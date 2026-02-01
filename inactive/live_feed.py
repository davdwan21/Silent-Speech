import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------
# Lip indices (order matters!)
# Your current lists = 40 points total
# -------------------------
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78]

LIP_ORDER = UPPER_LIPS + LOWER_LIPS
assert len(LIP_ORDER) == 40


# -------------------------
# Model (same as training)
# -------------------------
class GRUWordClassifier(nn.Module):
    def __init__(self, input_dim, hidden=128, num_classes=20):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)          # (B, T, 2H)
        pooled = out.mean(dim=1)      # (B, 2H)
        return self.head(pooled)

# -------------------------
# Feature extraction: 83 dims
# 80 = 40 points * (x,y)
# +3 extra scalars (placeholder-ish, but shape-correct)
# -------------------------
def extract_83_and_openness(landmarks_xy: np.ndarray):
    """
    returns: (feat83, openness)
      feat83: (83,) float32
      openness: float
    """
    pts = landmarks_xy[LIP_ORDER]  # (40,2)

    # Center by lip centroid
    center = pts.mean(axis=0, keepdims=True)
    pts_c = pts - center

    # Scale by mouth width
    left = landmarks_xy[61]
    right = landmarks_xy[291]
    mouth_w = np.linalg.norm(right - left) + 1e-6

    pts_n = pts_c / mouth_w
    feat80 = pts_n.reshape(-1).astype(np.float32)

    # --- openness ---
    openness = float(np.linalg.norm(landmarks_xy[13] - landmarks_xy[14]) / mouth_w)

    # --- 2 other extras to reach 83 (placeholders unless you match your dataset writer) ---
    height = float(np.linalg.norm(landmarks_xy[0] - landmarks_xy[17]) / mouth_w)
    corner = float(np.linalg.norm(landmarks_xy[61] - landmarks_xy[291]) / mouth_w) - 1.0

    extra3 = np.array([openness, height, corner], dtype=np.float32)  # (3,)
    feat83 = np.concatenate([feat80, extra3], axis=0)
    return feat83, openness

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


# -------------------------
# Drawing helpers
# -------------------------
def draw_lip_dots(frame_bgr, landmarks_xy, color=(0, 255, 0), radius=2):
    """Draw dots for LIP_ORDER points."""
    h, w = frame_bgr.shape[:2]
    for idx in LIP_ORDER:
        x = int(landmarks_xy[idx, 0] * w)
        y = int(landmarks_xy[idx, 1] * h)
        cv2.circle(frame_bgr, (x, y), radius, color, -1)


def draw_lip_outline(frame_bgr, landmarks_xy, color=(0, 255, 255), thickness=1):
    """Optional: draw a simple polyline through the lip points (in given order)."""
    h, w = frame_bgr.shape[:2]
    pts = []
    for idx in LIP_ORDER:
        x = int(landmarks_xy[idx, 0] * w)
        y = int(landmarks_xy[idx, 1] * h)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame_bgr, [pts], isClosed=True, color=color, thickness=thickness)


# -------------------------
# Main
# -------------------------
MODEL_CKPT = "word_model_resnet.pt"
FACE_TASK = "models/face_landmarker.task"
CAM_INDEX = 0

DRAW_POINTS = True
DRAW_OUTLINE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint
ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
id_to_label = ckpt["id_to_label"]
input_dim = int(ckpt["input_dim"])   # should be 83
max_t = int(ckpt["max_t"])
num_classes = len(id_to_label)

if input_dim != 83:
    raise ValueError(f"Your model expects input_dim={input_dim}, but this script is built for 83.")

model = GRUWordClassifier(input_dim=input_dim, hidden=128, num_classes=num_classes).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# MediaPipe FaceLandmarker
base_options = python.BaseOptions(model_asset_path=FACE_TASK)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# Rolling buffer
buf = deque(maxlen=max_t)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

PRED_EVERY = 2
WARMUP_MIN = min(10, max_t)

last_label = "..."
last_conf = 0.0

with torch.no_grad():
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            cv2.putText(frame_bgr, "No face found",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("live", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        lm = result.face_landmarks[0]
        landmarks_xy = np.array([(p.x, p.y) for p in lm], dtype=np.float32)  # (468,2)

        # Draw lip dots/outline
        if DRAW_POINTS:
            draw_lip_dots(frame_bgr, landmarks_xy, color=(0, 255, 0), radius=2)
        if DRAW_OUTLINE:
            draw_lip_outline(frame_bgr, landmarks_xy, color=(0, 255, 255), thickness=1)

        # Extract features and buffer
        feat, open_val = extract_83_and_openness(landmarks_xy)
        buf.append(feat)

        # Predict
        if len(buf) >= WARMUP_MIN and (frame_idx % PRED_EVERY == 0):
            X = np.zeros((max_t, input_dim), dtype=np.float32)
            seq = np.stack(list(buf), axis=0)  # (t,83)
            t = min(seq.shape[0], max_t)
            X[:t] = seq[:t]

            x_t = torch.from_numpy(X).unsqueeze(0).to(DEVICE)  # (1,T,D)
            logits = model(x_t).squeeze(0).cpu().numpy()       # (C,)
            probs = softmax_np(logits)

            pred_id = int(np.argmax(probs))
            last_label = str(id_to_label[pred_id])
            last_conf = float(np.max(probs))

        # Overlay text
        cv2.putText(frame_bgr, f"{last_label} ({last_conf:.2f})",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        if open_val > 0.04:
            cv2.putText(frame_bgr, f"Talking",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame_bgr, f"Not talking",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show
        cv2.imshow("live", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()