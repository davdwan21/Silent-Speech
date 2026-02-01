import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---- NEW: model deps
import torch
import torch.nn as nn

MODEL_PATH = "models/face_landmarker.task"
WORD_MODEL_PATH = "word_model_5.pt"   # 5-word model: hello, water, thanks, please, apple
CAM_INDEX = 0

DRAW_POINTS = True

LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78]

LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14

# Use same mouth landmark set as recorder
MOUTH_SET = sorted(set(UPPER_LIPS + LOWER_LIPS))

DEVICE = "cpu"


# ---------------- Model (same as training) ----------------
class GRUWordClassifier(nn.Module):
    def __init__(self, input_dim, hidden=64, num_classes=5):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True,
                          bidirectional=True)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)


# ---------------- Feature extraction (must match training) ----------------
def mouth_openness(face, w, h) -> float:
    def p(i):
        lm = face[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    open_dist = np.linalg.norm(p(UPPER_INNER) - p(LOWER_INNER))
    width = np.linalg.norm(p(LEFT_CORNER) - p(RIGHT_CORNER)) + 1e-6
    return float(open_dist / width)


def polygon_area_xy(pts_xy: np.ndarray) -> float:
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_mouth_features(face, w, h, prev_xy_norm=None):
    # Raw mouth XY in pixels
    xy = np.array([[face[i].x * w, face[i].y * h] for i in MOUTH_SET], dtype=np.float32)

    left = np.array([face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h], dtype=np.float32)
    right = np.array([face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h], dtype=np.float32)
    width = float(np.linalg.norm(left - right) + 1e-6)

    center = xy.mean(axis=0)
    xy_norm = (xy - center) / width

    openness = mouth_openness(face, w, h)

    # loop for area (normalize first so lip size matters less)
    loop_idx = UPPER_LIPS + LOWER_LIPS[::-1]
    loop_xy = np.array([[face[i].x * w, face[i].y * h] for i in loop_idx], dtype=np.float32)
    loop_xy_norm = (loop_xy - center) / width
    area = polygon_area_xy(loop_xy_norm)

    if prev_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_norm - prev_xy_norm, axis=1)))

    feat = np.concatenate(
        [xy_norm.reshape(-1), np.array([openness, area, vel], dtype=np.float32)],
        axis=0,
    )
    return feat, xy_norm


def pad_or_trim(X, max_t):
    T, D = X.shape
    if T >= max_t:
        return X[:max_t]
    out = np.zeros((max_t, D), dtype=np.float32)
    out[:T] = X
    return out


# ---------------- UI / drawing ----------------
def draw_lips_only(frame_bgr, result):
    out = frame_bgr.copy()
    if not result.face_landmarks:
        return out

    face = result.face_landmarks[0]
    h, w = out.shape[:2]

    def px(i):
        lm = face[i]
        return int(lm.x * w), int(lm.y * h)

    if DRAW_POINTS:
        for i in MOUTH_SET:
            cv2.circle(out, px(i), 1, (0, 255, 0), -1)

    return out


def open_camera():
    for idx in [CAM_INDEX, 1, 0]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened at index {idx}")
            return cap
        cap.release()
    raise RuntimeError("No camera found.")


# ---------------- Inference helpers ----------------
def load_word_model(path):
    ckpt = torch.load(path, map_location="cpu")
    input_dim = int(ckpt["input_dim"])
    max_t = int(ckpt["max_t"])
    id_to_label = ckpt["id_to_label"]
    num_classes = len(id_to_label)

    model = GRUWordClassifier(input_dim=input_dim, hidden=64, num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, id_to_label, max_t


def predict_clip(model, id_to_label, max_t, clip_feats):
    # clip_feats: list of (D,) float32
    if len(clip_feats) < 5:
        return None

    X = np.stack(clip_feats, axis=0).astype(np.float32)
    X = pad_or_trim(X, max_t)
    x = torch.from_numpy(X).unsqueeze(0)  # (1, T, D)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top3 = probs.argsort()[-3:][::-1]
    return [(id_to_label[int(i)], float(probs[i])) for i in top3]


def main():
    if not os.path.exists(WORD_MODEL_PATH):
        raise RuntimeError(f"Missing {WORD_MODEL_PATH}. Train first and create it.")

    word_model, id_to_label, max_t = load_word_model(WORD_MODEL_PATH)
    print("Loaded word model. max_t =", max_t)

    cap = open_camera()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cv2.namedWindow("Lips Only", cv2.WINDOW_NORMAL)

    recording = False
    clip_feats = []
    prev_xy_norm = None
    last_pred = None

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

            out = draw_lips_only(frame_bgr, result)

            # If recording, append features
            if recording and result.face_landmarks:
                face = result.face_landmarks[0]
                h, w = frame_bgr.shape[:2]
                feat, prev_xy_norm = extract_mouth_features(face, w, h, prev_xy_norm)
                clip_feats.append(feat)
            elif not recording:
                prev_xy_norm = None

            # Overlay
            status = "REC" if recording else "IDLE"
            cv2.putText(out, f"{status} | Press r start/stop | q quit",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if recording:
                cv2.putText(out, f"frames: {len(clip_feats)}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if last_pred:
                y0 = 110
                cv2.putText(out, "Top-3:",
                            (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for k, (lab, p) in enumerate(last_pred):
                    cv2.putText(out, f"{k+1}) {lab}  {p:.2f}",
                                (20, y0 + 30*(k+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Lips Only", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == ord("r"):
                recording = not recording
                if recording:
                    clip_feats = []
                    prev_xy_norm = None
                    last_pred = None
                    print("Recording started...")
                else:
                    print(f"Recording stopped. frames={len(clip_feats)}. Predicting...")
                    last_pred = predict_clip(word_model, id_to_label, max_t, clip_feats)
                    if last_pred:
                        print("Prediction:", last_pred)
                    else:
                        print("Clip too short / no landmarks.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
