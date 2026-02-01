# =============================
# FIXED LANDMARK INDEX RECORDER
# =============================

import os, time, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"
OUT_DIR = "clips_npz"
os.makedirs(OUT_DIR, exist_ok=True)

SPEAKER = "me"
CAM_INDEX = 1

ROI_W, ROI_H = 96, 48
SAVE_ROI = True
DRAW_POINTS = True

MOUTH_W_MIN_PX = 60
MOUTH_W_MAX_PX = 150

LEFT_CORNER = 61
RIGHT_CORNER = 291

# -----------------------------
# FIXED landmark sets
# -----------------------------
MOUTH_LOWER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95]
MOUTH_UPPER = [185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78]

CHIN_BOTTOM_ARC = [152,377,400,378,379,394,148,176,149,150,169]

CHEEKS = [
    214, 212, 57, 186, 202, 210, 204, 211, 194, 32,
    83, 201, 208, 18, 200, 199, 313, 421, 428, 396,
    406, 418, 262, 335, 424, 431, 273, 422, 430, 287,
    432, 434, 364, 410, 322, 436, 416
]


FIXED_IDXS = sorted(set(MOUTH_LOWER + MOUTH_UPPER + CHIN_BOTTOM_ARC + CHEEKS))
N_POINTS = len(FIXED_IDXS)

print(f"[INFO] Fixed landmarks: {N_POINTS} points")

# -----------------------------
# helpers
# -----------------------------
def mouth_width_px(face, w, h):
    L = np.array([face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h])
    R = np.array([face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h])
    return float(np.linalg.norm(L - R))


def extract_feature(face, w, h, idxs, prev_xy=None):
    """
    Points -> per-frame feature vector.

    Changes vs v3:
    - Normalize by mouth width (61-291) instead of subset width (prevents cheeks/jaw dominating scale)
    - Append speech scalars: [vel, mouth_open_px, mouth_w_px, mouth_aspect]
      where mouth_open uses inner lip midpoints (13-14).
    """
    # (K,2) pixel coords for selected points
    xy = np.array([[face[i].x * w, face[i].y * h] for i in idxs], np.float32)

    # center by subset mean (translation invariance)
    center = xy.mean(0)

    # scale by mouth width in pixels (stable speech anchor)
    mouth_w_px = mouth_width_px(face, w, h)
    scale = float(mouth_w_px + 1e-6)

    xy_n = (xy - center) / scale

    # simple velocity magnitude (reset prev_xy=None when you exit distance band)
    if prev_xy is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_n - prev_xy, axis=1)))

    # speech scalars (in pixels, plus aspect)
    upper = np.array([face[13].x * w, face[13].y * h], np.float32)
    lower = np.array([face[14].x * w, face[14].y * h], np.float32)
    mouth_open_px = float(np.linalg.norm(upper - lower))
    mouth_aspect = float(mouth_open_px / (mouth_w_px + 1e-6))

    feat = np.concatenate([
        xy_n.reshape(-1),
        np.array([vel, mouth_open_px, mouth_w_px, mouth_aspect], dtype=np.float32)
    ])
    return feat, xy_n, center, scale
def crop_roi(frame, center, scale):
    h, w = frame.shape[:2]
    cx, cy = center
    hw, hh = 1.2 * scale, 1.0 * scale
    x1, x2 = int(max(0, cx - hw)), int(min(w, cx + hw))
    y1, y2 = int(max(0, cy - hh)), int(min(h, cy + hh))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.resize(roi, (ROI_W, ROI_H))


def draw_points(frame, face, idxs):
    h, w = frame.shape[:2]
    for i in idxs:
        x, y = int(face[i].x * w), int(face[i].y * h)
        cv2.circle(frame, (x, y), 1, (0,255,0), -1)
    return frame

# -----------------------------
# main
# -----------------------------
def main():
    WORDS = ["yes","no","hello","thanks","please","fahhh","six","seven","lebron","aura"]
    KEYS = list("1234567890")
    key_to_word = dict(zip(KEYS, WORDS))

    cap = cv2.VideoCapture(CAM_INDEX)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    recording = False
    label = WORDS[0]
    bufX, bufT, bufR = [], [], []
    prev_xy = None
    clip_id = 0
    t0 = time.monotonic()

    with vision.FaceLandmarker.create_from_options(options) as lm:
        while True:
            ok, frame = cap.read()
            if not ok: break

            ts = int((time.monotonic() - t0) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(mp_img, ts)

            out = frame.copy()

            if res.face_landmarks:
                face = res.face_landmarks[0]
                h, w = frame.shape[:2]
                mw = mouth_width_px(face, w, h)
                in_range = MOUTH_W_MIN_PX <= mw <= MOUTH_W_MAX_PX

                if recording and in_range:
                    feat, prev_xy, center, scale = extract_feature(
                        face, w, h, FIXED_IDXS, prev_xy
                    )
                    bufX.append(feat)
                    bufT.append(ts)
                    if SAVE_ROI:
                        roi = crop_roi(frame, center, scale)
                        if roi is not None:
                            bufR.append(roi)
                else:
                    prev_xy = None

                if DRAW_POINTS:
                    draw_points(out, face, FIXED_IDXS)

                cv2.putText(out, f"mouth_w={mw:.1f}px", (20,140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if in_range else (0,0,255), 2)

            cv2.putText(out, f"{'REC' if recording else 'IDLE'} | {label}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            cv2.imshow("Recorder", out)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")): break
            ch = chr(key) if 32 <= key < 127 else ""
            if ch in key_to_word:
                label = key_to_word[ch]

            if ch == "r":
                recording = not recording
                if recording:
                    bufX, bufT, bufR = [], [], []
                    prev_xy = None
                else:
                    if len(bufX) > 5:
                        X = np.stack(bufX).astype(np.float32)
                        save = dict(
                            X=X,
                            ts=np.array(bufT),
                            label=label,
                            speaker=SPEAKER,
                            idxs=np.array(FIXED_IDXS),
                        )
                        if SAVE_ROI and bufR:
                            T = min(len(X), len(bufR))
                            save["X"] = X[:T]
                            save["roi"] = np.stack(bufR[:T])
                        fname = f"{SPEAKER}_{label}_{int(time.time())}_{clip_id:04d}.npz"
                        np.savez_compressed(os.path.join(OUT_DIR, fname), **save)
                        print("saved", fname)
                        clip_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()