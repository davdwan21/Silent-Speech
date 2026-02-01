import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/face_landmarker.task"
OUT_DIR = "lower_npz"
os.makedirs(OUT_DIR, exist_ok=True)

SPEAKER = "me"
SAVE_ROI = True
ROI_W, ROI_H = 96, 48
DRAW_POINTS = True

NOSE_TIP_INDEX = 1
LOWER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]
UPPER_LIPS = [
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
    311, 312, 13, 82, 81, 42, 183, 78
]

LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14

MOUTH_SET = sorted(set(UPPER_LIPS + LOWER_LIPS))
KEYS_10 = list("1234567890")

def get_lower_face_indices(face):
    """Pick indices below the nose tip based on normalized y."""
    nose_y = face[NOSE_TIP_INDEX].y
    return [i for i, lm in enumerate(face) if lm.y > nose_y]

def extract_lower_face_points(face, w, h, idxs, prev_xy_norm=None):
    """
    Returns:
      feat: flattened normalized xy points (len = 2 * len(idxs)) (+ optional vel)
      xy_norm: (N,2) normalized points
    """
    xy = np.array([[face[i].x * w, face[i].y * h] for i in idxs], dtype=np.float32)

    center = xy.mean(axis=0)

    # Scale by horizontal span (stable-ish for face size)
    width = float((xy[:, 0].max() - xy[:, 0].min()) + 1e-6)

    xy_norm = (xy - center) / width

    # Optional: velocity term (mean point motion)
    if prev_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_norm - prev_xy_norm, axis=1)))

    feat = np.concatenate([xy_norm.reshape(-1), np.array([vel], dtype=np.float32)], axis=0)
    return feat, xy_norm

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
    return float(
        0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    )


def extract_mouth_features(face, w, h, prev_xy_norm=None):
    xy = np.array(
        [[face[i].x * w, face[i].y * h] for i in MOUTH_SET],
        dtype=np.float32,
    )

    left = np.array(
        [face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h],
        dtype=np.float32,
    )
    right = np.array(
        [face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h],
        dtype=np.float32,
    )
    width = float(np.linalg.norm(left - right) + 1e-6)

    center = xy.mean(axis=0)
    xy_norm = (xy - center) / width

    openness = mouth_openness(face, w, h)

    loop_idx = UPPER_LIPS + LOWER_LIPS[::-1]
    loop_xy = np.array(
        [[face[i].x * w, face[i].y * h] for i in loop_idx],
        dtype=np.float32,
    )
    loop_xy_norm = (loop_xy - center) / width
    area = polygon_area_xy(loop_xy_norm)

    if prev_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_norm - prev_xy_norm, axis=1)))

    feat = np.concatenate(
        [
            xy_norm.reshape(-1),
            np.array([openness, area, vel], dtype=np.float32),
        ],
        axis=0,
    )

    return feat, xy_norm, center, width


def crop_mouth_roi_gray(frame_bgr, center_xy, mouth_width_px):
    h, w = frame_bgr.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])

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
    return roi


def draw_lower_face_only(frame_bgr, result, lower_face_idxs):
    out = frame_bgr.copy()
    if not result.face_landmarks:
        return out

    face = result.face_landmarks[0]
    h, w = out.shape[:2]

    # If we haven't locked indices yet, just show nothing (or everything, your choice)
    if lower_face_idxs is None:
        return out

    for i in lower_face_idxs:
        lm = face[i]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(out, (x, y), 1, (0, 255, 0), -1)

    return out



def main():
    WORDS = [
        "yes", "no", "hello", "thanks", "please",
        "fahhh", "six", "seven", "lebron", "aura",
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
    current_label = WORDS[0]
    buffer_X, buffer_ts, buffer_roi = [], [], []
    prev_xy_norm = None
    clip_id = 0
    lower_face_idxs = None
    prev_lower_xy_norm = None


    t0 = time.monotonic()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            ts_ms = int((time.monotonic() - t0) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_rgb
            )
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.face_landmarks:
                face = result.face_landmarks[0]
                if recording and lower_face_idxs is None:
                    lower_face_idxs = get_lower_face_indices(face)

            out = draw_lower_face_only(frame_bgr, result, lower_face_idxs)

            if recording and result.face_landmarks and lower_face_idxs is not None:
                face = result.face_landmarks[0]
                h, w = frame_bgr.shape[:2]

                feat, prev_lower_xy_norm = extract_lower_face_points(
                    face, w, h, lower_face_idxs, prev_xy_norm=prev_lower_xy_norm
                )

                buffer_X.append(feat)
                buffer_ts.append(ts_ms)


            status = "REC" if recording else "IDLE"
            cv2.putText(
                out,
                f"{status} | speaker: {SPEAKER} | label: {current_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

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
                    print("Recording started")
                    buffer_X, buffer_ts, buffer_roi = [], [], []
                    prev_xy_norm = None

                    # NEW: reset + re-lock per clip
                    lower_face_idxs = None
                    prev_lower_xy_norm = None
                else:
                    if len(buffer_X) > 5:
                        X = np.stack(buffer_X).astype(np.float32)
                        ts = np.array(buffer_ts, dtype=np.int32)

                        save_dict = dict(
                            X=X, ts=ts, label=current_label, speaker=SPEAKER
                        )

                        if SAVE_ROI and buffer_roi:
                            R = np.stack(buffer_roi).astype(np.uint8)
                            T = min(len(X), len(R))
                            save_dict["X"] = X[:T]
                            save_dict["ts"] = ts[:T]
                            save_dict["roi"] = R[:T]

                        fname = (
                            f"{SPEAKER}_{current_label}_"
                            f"{int(time.time())}_{clip_id:04d}.npz"
                        )
                        np.savez_compressed(
                            os.path.join(OUT_DIR, fname), **save_dict
                        )
                        clip_id += 1
                        print(f"Saved to {os.path.join(OUT_DIR, fname)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
