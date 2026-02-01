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

PT_PATH = "word_model.pt"      # your trained checkpoint
NPZ_DIR = "clips_npz"          # used only to read idxs + X shape once
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

# ----------------- MODEL (same as training) -----------------
class TemporalCNN(nn.Module):
    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        feat = self.net(x).squeeze(-1)
        return self.head(feat)

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ----------------- LOAD FEATURE SPEC FROM NPZ -----------------
def load_feature_spec(npz_dir: str):
    """
    Reads one npz to match training exactly:
      - idxs: which landmarks were used
      - d_npz: X.shape[1] (e.g., 345)
      - t_npz: X.shape[0] (e.g., 32)
    """
    paths = sorted(glob.glob(f"{npz_dir}/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")
    z = np.load(paths[0], allow_pickle=True)
    idxs = z["idxs"].astype(int).tolist()
    t_npz = int(z["X"].shape[0])
    d_npz = int(z["X"].shape[1])
    return idxs, t_npz, d_npz

def compute_openness(face, idxs):
    # robust scalar; likely close to what you used if you appended openness
    ys = [face[i].y for i in idxs]
    return float(max(ys) - min(ys))

def face_to_xvec(face, idxs, d_target, d_npz):
    """
    Build 1 frame feature vector to match your saved X format:
      base = [x,y for each idx in idxs] => 2*len(idxs)
      if d_npz == base+1, append openness as last feature
      then pad/truncate to d_target (from checkpoint)
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

    # Safety: coerce to d_target (should already match, but keep it robust)
    if feat.shape[0] > d_target:
        feat = feat[:d_target]
    elif feat.shape[0] < d_target:
        feat = np.pad(feat, (0, d_target - feat.shape[0]), mode="constant")

    return feat.astype(np.float32)

def pad_or_trim_time(X, T):
    # X: (t, D) -> (T, D)
    if X.shape[0] >= T:
        return X[:T]
    pad = np.zeros((T - X.shape[0], X.shape[1]), dtype=X.dtype)
    return np.vstack([X, pad])

def zscore_per_clip(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std

# ----------------- CLIP-GATED SETTINGS -----------------
# These are the knobs you tune live.
OPEN_THRESH = 0.15   # tune by looking at open=... overlay
START_N = 2           # frames above thresh to start "speaking"
END_N = 4             # frames below thresh to end clip
MAX_CLIP = 80         # safety cap on clip length
HOLD_FRAMES = 20      # keep last prediction on screen

CONF_THRESH = 0.45    # green overlay if >= this

def main():
    dev = get_device()

    # Load idxs + typical clip length + expected D from NPZ
    idxs, T_NPZ, D_NPZ = load_feature_spec(NPZ_DIR)
    print(f"NPZ spec: idxs={len(idxs)} | T~{T_NPZ} | D_npz={D_NPZ}")

    # Load checkpoint
    ckpt = torch.load(PT_PATH, map_location="cpu")
    D_TARGET = int(ckpt["d_in"])
    NUM_CLASSES = int(ckpt["num_classes"])
    id_to_word = ckpt.get("id_to_word", None)

    # Use training clip length if present, else use npz clip length
    T_TARGET = int(ckpt.get("t_target", T_NPZ))  # (if you never saved t_target, this becomes 32)

    print(f"Checkpoint: D_TARGET={D_TARGET} | classes={NUM_CLASSES} | T_TARGET={T_TARGET}")

    model = TemporalCNN(D_TARGET, NUM_CLASSES)
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

    # --- clip gating state ---
    speaking = False
    above_ct = 0
    below_ct = 0
    clip_buf = []  # list of xvec (D_TARGET,)

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

                # ---- draw your dots ----
                nose_base_y = max(face[i].y for i in NOSE_BOTTOM_FOR_CUTOFF)
                cut_y = nose_base_y + 0.003

                for idx, lm in enumerate(face):
                    if idx in NOSE_SET:
                        continue
                    if idx in CHEEK_SET or lm.y > cut_y:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(out, (x, y), DOT_RADIUS, DOT_COLOR, -1, lineType=cv2.LINE_AA)

                # ---- openness signal for gating ----
                openv = compute_openness(face, idxs)
                # Debug readout to tune threshold:
                cv2.putText(out, f"open={openv:.3f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                # update above/below counters
                if openv > OPEN_THRESH:
                    above_ct += 1
                    below_ct = 0
                else:
                    below_ct += 1
                    above_ct = 0

                # ---- gating logic ----
                if not speaking:
                    if above_ct >= START_N:
                        speaking = True
                        clip_buf = []
                        above_ct = 0
                        below_ct = 0
                else:
                    # collect features while speaking
                    xvec = face_to_xvec(face, idxs, D_TARGET, D_NPZ)
                    clip_buf.append(xvec)

                    # end conditions
                    if below_ct >= END_N or len(clip_buf) >= MAX_CLIP:
                        speaking = False
                        above_ct = 0
                        below_ct = 0

                        if len(clip_buf) >= 6:  # mirror your save condition len(buffer_X) > 5
                            Xclip = np.stack(clip_buf).astype(np.float32)   # (t, D)
                            Xclip = pad_or_trim_time(Xclip, T_TARGET)       # (T_TARGET, D)
                            Xclip = zscore_per_clip(Xclip)

                            x_t = torch.from_numpy(Xclip).unsqueeze(0).to(dev)  # (1,T,D)
                            with torch.no_grad():
                                logits = model(x_t)
                                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                                pred_id = int(probs.argmax())
                                conf = float(probs[pred_id])

                            last_conf = conf
                            if isinstance(id_to_word, dict):
                                last_pred = id_to_word.get(pred_id, str(pred_id))
                            else:
                                last_pred = str(pred_id)
                            hold = HOLD_FRAMES

                # ---- overlay prediction mask ----
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

                # speaking indicator
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
