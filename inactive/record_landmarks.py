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
WORD_MODEL_PATH = "word_model.pt"   # produced by train_words.py
CAM_INDEX = 0

SPEAKER = "me"          # <-- change if you want
SAVE_ROI = True         # <-- set False if you don't want mouth pixels yet
ROI_W, ROI_H = 96, 48   # small + CPU friendly (grayscale)

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
    def __init__(self, input_dim, hidden=128, num_classes=20):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.head(pooled)

MOUTH_SET = sorted(set(UPPER_LIPS + LOWER_LIPS))

# Map 20 labels to keys: 1-9,0,a-j
KEYS_10 = list("1234567890")


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

    openness = mouth_openness(face, w, h)

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
    feat = np.concatenate([xy_norm.reshape(-1),
                           np.array([openness, area, vel], dtype=np.float32)], axis=0)
    return feat, xy_norm, center, width

def crop_mouth_roi_gray(frame_bgr, center_xy, mouth_width_px):
    """
    Crop a mouth-aligned ROI centered at mouth centroid.
    Uses mouth width to scale crop size for stability.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])

    # crop box scales with mouth width
    half_w = 1.2 * mouth_width_px
    half_h = 0.8 * mouth_width_px

    x1 = int(max(0, cx - half_w))
    x2 = int(min(w, cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(h, cy + half_h))

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None

    roi = frame_bgr[y1:y2, x1:x2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (ROI_W, ROI_H), interpolation=cv2.INTER_AREA)
    return roi  # uint8 (H,W)


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
        cv2.putText(out, "No face found", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
        return out
    face = result.face_landmarks[0]
    h, w = out.shape[:2]

    def px(i):
        lm = face[i]
        return int(lm.x * w), int(lm.y * h)

    drawn = 0
    
    # Draw points
    if DRAW_POINTS:
        for i in MOUTH_SET:
            cv2.circle(out, px(i), 1, (255, 0, 0), -1)

    return out

def main():
    WORDS = [
        "yes","no","hello","thanks","please",
        "fahhh","six","seven","lebron","aura"
    ]
    key_to_word = {KEYS_10[i]: WORDS[i] for i in range(10)}

    cap = cv2.VideoCapture(1)

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
    current_label = WORDS[0]
    buffer_X, buffer_ts = [], []
    buffer_roi = []   # optional
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
                    buffer_X, buffer_ts, buffer_roi = [], [], []
                    prev_xy_norm = None
                    print(f"Recording started: {SPEAKER} / {current_label}")
                else:
                    # Save clip
                    if len(buffer_X) > 5:
                        X = np.stack(buffer_X, axis=0).astype(np.float32)
                        ts = np.array(buffer_ts, dtype=np.int32)

                        # ROI might have fewer frames if some crops failed; keep aligned length
                        save_dict = dict(X=X, ts=ts, label=current_label, speaker=SPEAKER)
                        if SAVE_ROI and len(buffer_roi) > 0:
                            R = np.stack(buffer_roi, axis=0).astype(np.uint8)  # (T,H,W)
                            T = min(len(X), len(R))
                            save_dict["X"] = X[:T]
                            save_dict["ts"] = ts[:T]
                            save_dict["roi"] = R[:T]  # uint8
                        fname = f"{SPEAKER}_{current_label}_{int(time.time())}_{clip_id:04d}.npz"
                        path = os.path.join(OUT_DIR, fname)
                        np.savez_compressed(path, **save_dict)

                        print(f"Saved {path} | frames={save_dict['X'].shape[0]} | dim={save_dict['X'].shape[1]}"
                              + (f" | roi={save_dict['roi'].shape}" if "roi" in save_dict else ""))

                        clip_id += 1
                    else:
                        print("Clip too short / no landmarks.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
