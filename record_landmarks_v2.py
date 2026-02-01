import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------- config ----------------
MODEL_PATH = "models/face_landmarker.task"
OUT_DIR = "clips_npz"
os.makedirs(OUT_DIR, exist_ok=True)

SPEAKER = "me"          # change if you want
SAVE_ROI = True         # also save grayscale mouth ROI frames
ROI_W, ROI_H = 96, 48   # must match training

DRAW_MOUTH_POINTS = True
DRAW_CHIN_POINTS = True

# Distance / scale gating (based on mouth width in pixels)
# Tune these after looking at the on-screen mouth_w value.
MOUTH_W_MIN_PX = 60
MOUTH_W_MAX_PX = 110

# ROI stabilization (EMA)
EMA_ALPHA = 0.25  # higher = follows quicker, lower = smoother

# ---------------- landmarks ----------------
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78]

# mouth anchors
LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14

MOUTH_SET = sorted(set(UPPER_LIPS + LOWER_LIPS))

# Chin / jawline-ish points (subset of FaceMesh face oval / jaw area)
# These help capture jaw motion + provide a quick quality check.
CHIN_SET = [
    152,              # chin tip
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,   # left jawline
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377  # right jawline
]
CHIN_SET = sorted(set(CHIN_SET))

# Map 10 labels to keys: 1-9,0
KEYS_10 = list("1234567890")

# ---------------- geometry helpers ----------------
def _px(face, i, w, h):
    lm = face[i]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def mouth_openness(face, w, h) -> float:
    open_dist = np.linalg.norm(_px(face, UPPER_INNER, w, h) - _px(face, LOWER_INNER, w, h))
    width = np.linalg.norm(_px(face, LEFT_CORNER, w, h) - _px(face, RIGHT_CORNER, w, h)) + 1e-6
    return float(open_dist / width)

def polygon_area_xy(pts_xy: np.ndarray) -> float:
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def ema_update(prev, cur, alpha):
    if prev is None:
        return cur
    return (1.0 - alpha) * prev + alpha * cur

def extract_features_with_chin(face, w, h, prev_mouth_xy_norm=None):
    """
    Returns:
      feat: (D,) float32
      mouth_xy_norm: (K,2)
      mouth_center_px: (2,)
      mouth_w_px: float
    """
    # ---- mouth ----
    mouth_xy = np.stack([_px(face, i, w, h) for i in MOUTH_SET], axis=0)  # (Km,2)
    left = _px(face, LEFT_CORNER, w, h)
    right = _px(face, RIGHT_CORNER, w, h)
    mouth_w = float(np.linalg.norm(left - right) + 1e-6)
    mouth_center = mouth_xy.mean(axis=0)

    mouth_xy_norm = (mouth_xy - mouth_center) / mouth_w

    # openness/area/vel in normalized space
    openness = mouth_openness(face, w, h)

    loop_idx = UPPER_LIPS + LOWER_LIPS[::-1]
    loop_xy = np.stack([_px(face, i, w, h) for i in loop_idx], axis=0)
    loop_xy_norm = (loop_xy - mouth_center) / mouth_w
    area = polygon_area_xy(loop_xy_norm)

    if prev_mouth_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(mouth_xy_norm - prev_mouth_xy_norm, axis=1)))

    # ---- chin / jaw (normalized by mouth width, centered at mouth center) ----
    chin_xy = np.stack([_px(face, i, w, h) for i in CHIN_SET], axis=0)  # (Kc,2)
    chin_xy_norm = (chin_xy - mouth_center) / mouth_w

    # Feature vector: [mouth_xy_norm_flat, chin_xy_norm_flat, openness, area, vel]
    feat = np.concatenate(
        [
            mouth_xy_norm.reshape(-1),
            chin_xy_norm.reshape(-1),
            np.array([openness, area, vel], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    return feat, mouth_xy_norm, mouth_center, mouth_w

def crop_mouth_roi_gray(frame_bgr, center_xy, mouth_width_px):
    """
    Crop a stabilized mouth ROI centered at `center_xy`, scale by `mouth_width_px`.
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

def draw_points(frame_bgr, face):
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    def p(i):
        lm = face[i]
        return int(lm.x * w), int(lm.y * h)

    if DRAW_MOUTH_POINTS:
        for i in MOUTH_SET:
            cv2.circle(out, p(i), 1, (255, 0, 0), -1)

    if DRAW_CHIN_POINTS:
        for i in CHIN_SET:
            cv2.circle(out, p(i), 1, (0, 165, 255), -1)  # orange

    return out

# ---------------- main ----------------
def main():
    # Edit your labels here
    WORDS = [
        "yes", "no", "hello", "thanks", "please",
        "fahhh", "six", "seven", "lebron", "aura"
    ]
    key_to_word = {KEYS_10[i]: WORDS[i] for i in range(len(WORDS))}

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera index 1. Try 0.")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cv2.namedWindow("Recorder", cv2.WINDOW_NORMAL)

    recording = False
    current_label = WORDS[0]

    buffer_X, buffer_ts, buffer_roi = [], [], []
    prev_mouth_xy_norm = None
    clip_id = 0

    # ROI stabilization state
    ema_center = None
    ema_mouth_w = None

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

            have_face = bool(result.face_landmarks)
            in_range = False
            mouth_w_px = None

            if have_face:
                face = result.face_landmarks[0]
                out = draw_points(out, face)

                h, w = frame_bgr.shape[:2]
                feat, mouth_xy_norm, mouth_center, mouth_w = extract_features_with_chin(
                    face, w, h, prev_mouth_xy_norm
                )
                prev_mouth_xy_norm = mouth_xy_norm

                mouth_w_px = mouth_w
                in_range = (MOUTH_W_MIN_PX <= mouth_w_px <= MOUTH_W_MAX_PX)

                # update EMA stabilization (even if not recording, so it settles)
                ema_center = ema_update(ema_center, mouth_center, EMA_ALPHA)
                ema_mouth_w = ema_update(ema_mouth_w, np.array([mouth_w], dtype=np.float32), EMA_ALPHA)

                # record only if in-range (prevents "too close / too far" clips)
                if recording and in_range:
                    buffer_X.append(feat)
                    buffer_ts.append(ts_ms)

                    if SAVE_ROI and ema_center is not None and ema_mouth_w is not None:
                        roi = crop_mouth_roi_gray(frame_bgr, ema_center, float(ema_mouth_w[0]))
                        if roi is not None:
                            buffer_roi.append(roi)

            else:
                prev_mouth_xy_norm = None
                ema_center = None
                ema_mouth_w = None

            # ---------------- UI ----------------
            status = "REC" if recording else "IDLE"
            cv2.putText(out, f"{status} | speaker: {SPEAKER} | label: {current_label}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2)

            cv2.putText(out, "Keys: r start/stop | 1-0 set label | q quit",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

            # range indicator
            if mouth_w_px is None:
                cv2.putText(out, "Face: NOT DETECTED", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            else:
                color = (0, 255, 0) if in_range else (0, 0, 255)
                cv2.putText(out, f"mouth_w(px): {mouth_w_px:.1f}  target [{MOUTH_W_MIN_PX},{MOUTH_W_MAX_PX}]",
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if recording and not in_range:
                    cv2.putText(out, "OUT OF RANGE: not recording frames",
                                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if recording:
                cv2.putText(out, f"frames_saved: {len(buffer_X)}", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # show ROI preview (stabilized) in corner
            if SAVE_ROI and ema_center is not None and ema_mouth_w is not None:
                preview = crop_mouth_roi_gray(frame_bgr, ema_center, float(ema_mouth_w[0]))
                if preview is not None:
                    ph, pw = preview.shape[:2]
                    # place in top-right
                    x0 = out.shape[1] - pw - 20
                    y0 = 20
                    out[y0:y0+ph, x0:x0+pw] = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

            cv2.imshow("Recorder", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            ch = chr(key) if 32 <= key < 127 else ""
            if ch in key_to_word:
                current_label = key_to_word[ch]

            if ch == "r":
                recording = not recording
                if recording:
                    buffer_X, buffer_ts, buffer_roi = [], [], []
                    prev_mouth_xy_norm = None
                    print(f"Recording started: {SPEAKER} / {current_label}")
                else:
                    # save
                    if len(buffer_X) > 5:
                        X = np.stack(buffer_X, axis=0).astype(np.float32)
                        ts = np.array(buffer_ts, dtype=np.int32)

                        save_dict = dict(X=X, ts=ts, label=current_label, speaker=SPEAKER)

                        if SAVE_ROI and len(buffer_roi) > 0:
                            R = np.stack(buffer_roi, axis=0).astype(np.uint8)  # (T,H,W)
                            T = min(len(X), len(R))
                            save_dict["X"] = X[:T]
                            save_dict["ts"] = ts[:T]
                            save_dict["roi"] = R[:T]

                        fname = f"{SPEAKER}_{current_label}_{int(time.time())}_{clip_id:04d}.npz"
                        path = os.path.join(OUT_DIR, fname)
                        np.savez_compressed(path, **save_dict)

                        print(f"Saved {path} | frames={save_dict['X'].shape[0]} | dim={save_dict['X'].shape[1]}"
                              + (f" | roi={save_dict['roi'].shape}" if "roi" in save_dict else ""))

                        clip_id += 1
                    else:
                        print("Clip too short / not enough in-range frames; not saved.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
