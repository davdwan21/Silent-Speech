import time
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"
CAM_INDEX = 1

PT_PATH = "word_model.pt"
NPZ_DIR = "clips_npz"
WINDOW_NAME = "Live + Word Prediction (Clip Gated)"

DOT_RADIUS = 1
DOT_COLOR = (0, 255, 0)

# ----------------- YOUR SETS -----------------
NOSE_SET = {
    1, 2, 4, 5, 6, 19, 20,
    168, 197, 195, 193, 122, 196, 3,
    45, 44, 48, 49, 51, 52, 53,
    275, 274, 278, 279, 281, 282, 283,
    114, 115, 131, 134, 102,
    343, 344, 360, 363, 331,
    94, 97, 99, 100, 101,
    328, 326, 327, 294, 305
}
NOSE_BOTTOM_FOR_CUTOFF = [2, 94, 97, 328, 326]

LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148]
CHEEK_SET = set(LEFT_CHEEK + RIGHT_CHEEK)
CHEEK_EXPAND = 0

def expand_by_index_neighbors(idx_set, k=1):
    if k <= 0:
        return set(idx_set)
    out = set(idx_set)
    for _ in range(k):
        more = set()
        for i in out:
            for j in (i - 1, i + 1, i - 2, i + 2):
                if 0 <= j < 468:
                    more.add(j)
        out |= more
    return out

CHEEK_SET = expand_by_index_neighbors(CHEEK_SET, CHEEK_EXPAND)

# ----------------- DEVICE -----------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ----------------- MODEL (MATCH TRAINING) -----------------
class GRUClassifier(nn.Module):
    def __init__(self, d_in: int, num_classes: int, hidden=128, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        _, h = self.gru(x)      # h: (1,B,H)
        h = self.drop(h[0])     # (B,H)
        return self.head(h)

# ----------------- LOAD FEATURE SPEC FROM NPZ -----------------
def load_feature_spec(npz_dir: str):
    paths = sorted(glob.glob(f"{npz_dir}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")
    z = np.load(paths[0], allow_pickle=True)
    idxs = z["idxs"].astype(int).tolist()
    t_npz = int(z["X"].shape[0])
    d_npz = int(z["X"].shape[1])
    return idxs, t_npz, d_npz

def compute_openness(face, idxs):
    ys = [face[i].y for i in idxs]
    return float(max(ys) - min(ys))

def face_to_xvec(face, idxs, d_npz):
    """
    Produce the same per-frame vector layout as recording:
      [x0,y0,x1,y1,...] and possibly + openness if D_npz == 2*len(idxs)+1
    Returns length D_npz (NOT d_target).
    """
    base = np.empty((2 * len(idxs),), dtype=np.float32)
    j = 0
    for i in idxs:
        lm = face[i]
        base[j] = lm.x
        base[j + 1] = lm.y
        j += 2

    if d_npz == base.shape[0] + 1:
        openv = compute_openness(face, idxs)
        feat = np.concatenate([base, np.array([openv], dtype=np.float32)], axis=0)
    else:
        feat = base

    return feat.astype(np.float32)

# ----------------- TRAINING-MATCH PREPROCESS -----------------
def fix_dim(X: np.ndarray, d_target: int) -> np.ndarray:
    # X: (T, D_var) -> (T, d_target)
    D = X.shape[1]
    if D == d_target:
        return X
    if D > d_target:
        return X[:, :d_target]
    pad = np.zeros((X.shape[0], d_target - D), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)

def pad_or_trim_time(X: np.ndarray, T: int) -> np.ndarray:
    if X.shape[0] >= T:
        return X[:T]
    pad = np.zeros((T - X.shape[0], X.shape[1]), dtype=X.dtype)
    return np.vstack([X, pad])

def zscore_per_clip(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std

def activity_from_X(X: np.ndarray) -> np.ndarray:
    # if openness scalar exists (odd D), use it
    if X.shape[1] % 2 == 1:
        return X[:, -1].astype(np.float32)
    y = X[:, 1::2]
    return (y.max(axis=1) - y.min(axis=1)).astype(np.float32)

def trim_clip_like_training(X: np.ndarray, T_target: int, q=0.60, margin=2, min_keep=6) -> np.ndarray:
    a = activity_from_X(X)
    thr = float(np.quantile(a, q))
    active = np.where(a > thr)[0]
    if len(active) < min_keep:
        return pad_or_trim_time(X, T_target)
    lo = max(int(active[0]) - margin, 0)
    hi = min(int(active[-1]) + margin + 1, X.shape[0])
    return pad_or_trim_time(X[lo:hi], T_target)

def add_deltas(X: np.ndarray) -> np.ndarray:
    dX = np.zeros_like(X)
    dX[1:] = X[1:] - X[:-1]
    return np.concatenate([X, dX], axis=1)

def id_to_label(pred_id, id_to_word):
    if id_to_word is None:
        return str(pred_id)
    if isinstance(id_to_word, (list, tuple)):
        return id_to_word[pred_id] if 0 <= pred_id < len(id_to_word) else str(pred_id)
    if isinstance(id_to_word, dict):
        if pred_id in id_to_word:
            return id_to_word[pred_id]
        if str(pred_id) in id_to_word:
            return id_to_word[str(pred_id)]
    return str(pred_id)

# ----------------- CLIP-GATED SETTINGS -----------------
OPEN_THRESH = 0.15    # start here; tune using open=... overlay
START_N = 2
END_N = 4
MAX_CLIP = 80
HOLD_FRAMES = 20

CONF_THRESH = 0.45

def main():
    dev = get_device()

    # Load idxs + NPZ D (for per-frame vector layout)
    idxs, T_NPZ, D_NPZ = load_feature_spec(NPZ_DIR)
    print(f"NPZ spec: idxs={len(idxs)} | T~{T_NPZ} | D_npz={D_NPZ}")

    ckpt = torch.load(PT_PATH, map_location="cpu")
    NUM_CLASSES = int(ckpt["num_classes"])
    id_to_word = ckpt.get("id_to_word", None)

    # training-time params
    T_TARGET = int(ckpt.get("t_target", T_NPZ))
    D_TARGET_BASE = int(ckpt.get("d_target", D_NPZ))   # base D before deltas
    USE_DELTAS = bool(ckpt.get("use_deltas", False))

    # model input dim (after deltas if used)
    D_IN = int(ckpt["d_in"])

    trim_cfg = ckpt.get("trim", {"q": 0.60, "margin": 2, "min_keep": 6})
    TRIM_Q = float(trim_cfg.get("q", 0.60))
    TRIM_MARGIN = int(trim_cfg.get("margin", 2))
    TRIM_MIN_KEEP = int(trim_cfg.get("min_keep", 6))

    print(f"Checkpoint: D_IN={D_IN} | classes={NUM_CLASSES} | T_TARGET={T_TARGET} | "
          f"D_TARGET_BASE={D_TARGET_BASE} | use_deltas={USE_DELTAS} | trim={trim_cfg}")

    model = GRUClassifier(d_in=D_IN, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state"])
    model.to(dev).eval()

    cap = cv2.VideoCapture(CAM_INDEX)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    speaking = False
    above_ct = 0
    below_ct = 0
    clip_buf = []

    last_pred = None
    last_conf = 0.0
    hold = 0

    t0 = time.monotonic()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts_ms = int((time.monotonic() - t0) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            out = frame_bgr.copy()

            if result.face_landmarks:
                face = result.face_landmarks[0]
                h, w = out.shape[:2]

                # ---- draw dots ----
                nose_base_y = max(face[i].y for i in NOSE_BOTTOM_FOR_CUTOFF)
                cut_y = nose_base_y + 0.003

                for idx, lm in enumerate(face):
                    if idx in NOSE_SET:
                        continue
                    if idx in CHEEK_SET or lm.y > cut_y:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(out, (x, y), DOT_RADIUS, DOT_COLOR, -1, lineType=cv2.LINE_AA)

                # ---- openness for gating ----
                openv = compute_openness(face, idxs)
                cv2.putText(out, f"open={openv:.3f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                if openv > OPEN_THRESH:
                    above_ct += 1
                    below_ct = 0
                else:
                    below_ct += 1
                    above_ct = 0

                if not speaking:
                    if above_ct >= START_N:
                        speaking = True
                        clip_buf = []
                        above_ct = 0
                        below_ct = 0
                else:
                    xvec = face_to_xvec(face, idxs, D_NPZ)  # length D_NPZ
                    clip_buf.append(xvec)

                    if below_ct >= END_N or len(clip_buf) >= MAX_CLIP:
                        speaking = False
                        above_ct = 0
                        below_ct = 0

                        if len(clip_buf) >= 6:
                            Xclip = np.stack(clip_buf).astype(np.float32)      # (t, D_NPZ)

                            # ---- MATCH TRAINING PIPELINE ----
                            Xclip = fix_dim(Xclip, D_TARGET_BASE)              # (t, d_target_base)
                            Xclip = trim_clip_like_training(
                                Xclip, T_TARGET, q=TRIM_Q, margin=TRIM_MARGIN, min_keep=TRIM_MIN_KEEP
                            )                                                   # (T_TARGET, d_target_base)
                            Xclip = zscore_per_clip(Xclip)

                            if USE_DELTAS:
                                Xclip = add_deltas(Xclip)                      # (T_TARGET, 2*d_target_base)
                                Xclip = zscore_per_clip(Xclip)

                            # Ensure final dim matches model input
                            if Xclip.shape[1] != D_IN:
                                # last-resort safety (shouldn't happen)
                                Xclip = fix_dim(Xclip, D_IN)

                            x_t = torch.from_numpy(Xclip).unsqueeze(0).to(dev)  # (1,T,D_IN)
                            with torch.no_grad():
                                logits = model(x_t)
                                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                                pred_id = int(probs.argmax())
                                conf = float(probs[pred_id])

                            last_conf = conf
                            last_pred = id_to_label(pred_id, id_to_word)
                            hold = HOLD_FRAMES

                # overlay
                if hold > 0 and last_pred is not None:
                    hold -= 1
                    txt = f"PRED: {last_pred} ({last_conf:.2f})"
                    overlay_color = (0, 255, 0) if last_conf >= CONF_THRESH else (0, 0, 255)
                    alpha = 0.18
                    overlay = out.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 60), overlay_color, -1)
                    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
                    cv2.putText(out, txt, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(out, "SPEAKING" if speaking else "IDLE", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255) if speaking else (200, 200, 200), 2, cv2.LINE_AA)

            else:
                cv2.putText(out, "NO FACE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                speaking = False
                above_ct = below_ct = 0
                clip_buf = []
                last_pred = None
                last_conf = 0.0
                hold = 0

            cv2.putText(out, "q to quit", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
