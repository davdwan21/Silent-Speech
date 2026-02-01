# live_word_predict_mlp.py
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
CAM_INDEX = 0

PT_PATH = "word_model.pt"
NPZ_DIR = "clips_npz"

WINDOW_NAME = "Live + Word Prediction (Clip Gated) — MLP"

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

# ----------------- MODEL (MLP: mean+std features) -----------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

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
    # raw coords for the exact training indices
    xs = np.array([face[i].x for i in idxs], dtype=np.float32)
    ys = np.array([face[i].y for i in idxs], dtype=np.float32)

    # center per frame (very likely what your recorder did)
    xs = xs - xs.mean()
    ys = ys - ys.mean()

    base = np.empty((2 * len(idxs),), dtype=np.float32)
    base[0::2] = xs
    base[1::2] = ys

    if d_npz == base.shape[0] + 1:
        openv = float(ys.max() - ys.min())
        return np.concatenate([base, np.array([openv], dtype=np.float32)], axis=0).astype(np.float32)

    return base.astype(np.float32)


# ----------------- PREPROCESS (match quick MLP training) -----------------
def fix_dim_1d(x: np.ndarray, d_target: int) -> np.ndarray:
    # x: (D,) -> (d_target,)
    D = x.shape[0]
    if D == d_target:
        return x
    if D > d_target:
        return x[:d_target]
    pad = np.zeros((d_target - D,), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)

def clip_to_feat_mean_std(X: np.ndarray) -> np.ndarray:
    # X: (t, D)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)  # (2D,)

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
OPEN_THRESH = 0.18   # was 0.15
START_N = 3          # was 2
END_N = 5            # was 4
MAX_CLIP = 60        # a bit shorter is fine
HOLD_FRAMES = 20

CONF_THRESH = 0.45

def main():
    dev = get_device()

    # Load idxs + NPZ D (for per-frame vector layout)
    idxs, T_NPZ, D_NPZ = load_feature_spec(NPZ_DIR)
    print(f"NPZ spec: idxs={len(idxs)} | T~{T_NPZ} | D_npz={D_NPZ}")

    # Load quick MLP checkpoint produced by train_quick_words.py
    ckpt = torch.load(PT_PATH, map_location="cpu")
    labels = ckpt["labels"]            # list: ["hello","yes","no","please","thanks"]
    D_IN = int(ckpt["in_dim"])         # should be 2*D_NPZ typically
    NUM_CLASSES = len(labels)

    expected = 2 * D_NPZ
    if D_IN != expected:
        print(f"[WARN] ckpt in_dim={D_IN} but 2*D_npz={expected}. "
              f"This can still run, but you probably want them to match.")

    model = MLP(in_dim=D_IN, num_classes=NUM_CLASSES)
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
                    xvec = face_to_xvec(face, idxs, D_NPZ)  # (D_NPZ,)
                    clip_buf.append(xvec)

                    if below_ct >= END_N or len(clip_buf) >= MAX_CLIP:
                        speaking = False
                        above_ct = 0
                        below_ct = 0

                        # run prediction if we have enough frames
                        if len(clip_buf) >= 6:
                            Xclip = np.stack(clip_buf).astype(np.float32)   # (t, D_NPZ)

                            # match quick training: mean+std over time
                            feat = clip_to_feat_mean_std(Xclip)             # (2*D_NPZ,)
                            feat = fix_dim_1d(feat, D_IN)                   # (D_IN,)

                            x_t = torch.from_numpy(feat).unsqueeze(0).to(dev)  # (1, D_IN)
                            with torch.no_grad():
                                logits = model(x_t)
                                probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                                pred_id = int(probs.argmax())
                                conf = float(probs[pred_id])

                            last_conf = conf
                            last_pred = id_to_label(pred_id, labels)
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
