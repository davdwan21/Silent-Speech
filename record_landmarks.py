import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"
OUT_DIR = "clips_npz"
os.makedirs(OUT_DIR, exist_ok=True)

DRAW_POINTS = True

LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78]

LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14

MOUTH_SET = sorted(set(UPPER_LIPS + LOWER_LIPS))  # unique indices

# Map 20 labels to keys: 1-9,0,a-j
KEYS_20 = list("1234567890abcdefghij")

def mouth_openness(face, w, h) -> float:
    def p(i):
        lm = face[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    open_dist = np.linalg.norm(p(UPPER_INNER) - p(LOWER_INNER))
    width = np.linalg.norm(p(LEFT_CORNER) - p(RIGHT_CORNER)) + 1e-6
    return float(open_dist / width)

def polygon_area_xy(pts_xy: np.ndarray) -> float:
    # pts_xy: (N,2), assumed ordered around boundary
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def extract_mouth_features(face, w, h, prev_xy_norm=None):
    # Raw mouth XY in pixels
    xy = np.array([[face[i].x * w, face[i].y * h] for i in MOUTH_SET], dtype=np.float32)  # (K,2)

    # Anchors in pixels
    left = np.array([face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h], dtype=np.float32)
    right = np.array([face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h], dtype=np.float32)

    width = float(np.linalg.norm(left - right) + 1e-6)

    # Normalize: center by mouth centroid, scale by mouth width
    center = xy.mean(axis=0)
    xy_norm = (xy - center) / width  # (K,2)

    # Derived features (scale-invariant now)
    openness = mouth_openness(face, w, h)  # already width-normalized
    # Area using the outer-ish loop: use UPPER then reversed LOWER for a loop
    loop_idx = UPPER_LIPS + LOWER_LIPS[::-1]
    loop_xy = np.array([[face[i].x * w, face[i].y * h] for i in loop_idx], dtype=np.float32)
    loop_xy_norm = (loop_xy - center) / width
    area = polygon_area_xy(loop_xy_norm)

    # Motion (mean point speed in normalized space)
    if prev_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_norm - prev_xy_norm, axis=1)))

    # Feature vector: flattened landmarks + [openness, area, vel]
    feat = np.concatenate([xy_norm.reshape(-1), np.array([openness, area, vel], dtype=np.float32)], axis=0)
    return feat, xy_norm

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
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened at index {idx}")
            return cap
        cap.release()
    raise RuntimeError("No camera found at index 0 or 1")

def main():
    # Put your 20 words here in the same order as KEYS_20
    WORDS = [
        "yes","no","hello","thanks","please",
        "stop","go","left","right","up",
        "down","one","two","three","four",
        "five","water","help","sorry","okay"
    ]
    key_to_word = {KEYS_20[i]: WORDS[i] for i in range(20)}

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
    current_label = WORDS[0]
    buffer_X = []
    buffer_ts = []
    prev_xy_norm = None
    clip_id = 0

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

            # UI overlay
            status = "REC" if recording else "IDLE"
            cv2.putText(out, f"{status} | label: {current_label}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(out, "Keys: r start/stop | 1-0,a-j set label | q quit",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Log features if recording & landmarks present
            if recording and result.face_landmarks:
                face = result.face_landmarks[0]
                h, w = frame_bgr.shape[:2]
                feat, prev_xy_norm = extract_mouth_features(face, w, h, prev_xy_norm)
                buffer_X.append(feat)
                buffer_ts.append(ts_ms)
            elif not recording:
                prev_xy_norm = None  # reset motion baseline when not recording

            cv2.imshow("Lips Only", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            ch = chr(key) if 32 <= key < 127 else ""
            if ch in key_to_word:
                current_label = key_to_word[ch]

            if ch == "r":
                recording = not recording
                if recording:
                    buffer_X, buffer_ts = [], []
                    prev_xy_norm = None
                    print(f"Recording started: {current_label}")
                else:
                    # Save clip
                    if len(buffer_X) > 5:
                        X = np.stack(buffer_X, axis=0).astype(np.float32)
                        ts = np.array(buffer_ts, dtype=np.int32)
                        fname = f"{current_label}_{int(time.time())}_{clip_id:04d}.npz"
                        path = os.path.join(OUT_DIR, fname)
                        np.savez_compressed(path, X=X, ts=ts, label=current_label)
                        print(f"Saved {path} | frames={len(X)} | dim={X.shape[1]}")
                        clip_id += 1
                    else:
                        print("Clip too short; not saved.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
