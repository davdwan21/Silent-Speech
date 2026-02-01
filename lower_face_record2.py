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

SPEAKER = "me"
SAVE_ROI = True
ROI_W, ROI_H = 96, 48
DRAW_POINTS = True
CAM_INDEX = 1

# Distance / scale gating (based on mouth width in pixels)
# Tune these after looking at the on-screen mouth_w value.
MOUTH_W_MIN_PX = 80
MOUTH_W_MAX_PX = 110

KEYS_10 = list("1234567890")

# -----------------------------
# Your index sets (as provided)
# -----------------------------
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

CHEEK_EXPAND = 0  # set to 1 or 2 if desired


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


# -----------------------------
# Selection + feature extraction
# -----------------------------
CUT_MARGIN = 0.003  # same as your viewer script

# Mouth anchors (MediaPipe Face Mesh)
LEFT_CORNER = 61
RIGHT_CORNER = 291


def mouth_width_px(face, w, h) -> float:
    """Mouth width in pixels (proxy for camera distance)."""
    left = np.array([face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h], dtype=np.float32)
    right = np.array([face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h], dtype=np.float32)
    return float(np.linalg.norm(left - right))

def compute_selected_indices(face):
    """
    Returns a fixed ordered list of landmark indices to record for this clip,
    based on your rule:
      include if (idx in CHEEK_SET) OR (lm.y > cut_y)
      but exclude if idx in NOSE_SET
    """
    nose_base_y = max(face[i].y for i in NOSE_BOTTOM_FOR_CUTOFF)
    cut_y = nose_base_y + CUT_MARGIN

    selected = []
    for idx, lm in enumerate(face):
        if idx in NOSE_SET:
            continue
        if (idx in CHEEK_SET) or (lm.y > cut_y):
            selected.append(idx)

    # make ordering stable
    selected.sort()
    return selected


def extract_points_feature(face, w, h, idxs, prev_xy_norm=None, add_vel=True):
    """
    - Takes only idxs
    - Converts to pixel xy
    - Normalizes (centered / scale by horizontal span)
    - Returns flattened vector (+ optional velocity)
    """
    xy = np.array([[face[i].x * w, face[i].y * h] for i in idxs], dtype=np.float32)

    center = xy.mean(axis=0)
    width = float((xy[:, 0].max() - xy[:, 0].min()) + 1e-6)  # scale by face width in this subset
    xy_norm = (xy - center) / width

    if not add_vel:
        return xy_norm.reshape(-1), xy_norm, center, width

    if prev_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_norm - prev_xy_norm, axis=1)))

    feat = np.concatenate([xy_norm.reshape(-1), np.array([vel], dtype=np.float32)], axis=0)
    return feat, xy_norm, center, width


def crop_roi_gray(frame_bgr, center_xy, width_px):
    """
    ROI around the LOWER FACE subset center. If you prefer mouth-only ROI,
    we can switch this back to your mouth crop logic.
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])

    half_w = 1.2 * width_px
    half_h = 1.0 * width_px  # slightly taller since it's lower-face-ish

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


def draw_selected(frame_bgr, face, idxs):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for i in idxs:
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

    cap = cv2.VideoCapture(CAM_INDEX)

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

    selected_idxs = None  # locked per clip
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

            out = frame_bgr.copy()

            if result.face_landmarks:
                face = result.face_landmarks[0]
                h, w = out.shape[:2]

                # ---- Distance gating (proxy: mouth width in pixels)
                mouth_w = mouth_width_px(face, w, h)
                in_range = (MOUTH_W_MIN_PX <= mouth_w <= MOUTH_W_MAX_PX)

                if mouth_w < MOUTH_W_MIN_PX:
                    dist_msg = "TOO FAR"
                    dist_color = (0, 0, 255)
                elif mouth_w > MOUTH_W_MAX_PX:
                    dist_msg = "TOO CLOSE"
                    dist_color = (0, 0, 255)
                else:
                    dist_msg = "DIST OK"
                    dist_color = (0, 255, 0)

                # lock indices once per clip (first good face during recording AND in-range)
                if recording and selected_idxs is None and in_range:
                    selected_idxs = compute_selected_indices(face)

                if selected_idxs is not None and DRAW_POINTS:
                    out = draw_selected(frame_bgr, face, selected_idxs)

                # record datapoints (only when in the allowed distance band)
                if recording and selected_idxs is not None and in_range:
                    feat, prev_xy_norm, center, width_px = extract_points_feature(
                        face, w, h, selected_idxs, prev_xy_norm=prev_xy_norm, add_vel=True
                    )
                    buffer_X.append(feat)
                    buffer_ts.append(ts_ms)

                    if SAVE_ROI:
                        roi = crop_roi_gray(frame_bgr, center, width_px)
                        if roi is not None:
                            buffer_roi.append(roi)

                # If we're recording but out of range, avoid a big velocity jump when re-entering range
                if recording and (not in_range):
                    prev_xy_norm = None

                cv2.putText(out, f"mouth_w={mouth_w:.1f}px | {dist_msg}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, dist_color, 2, cv2.LINE_AA)
                cv2.putText(out, f"range [{MOUTH_W_MIN_PX}, {MOUTH_W_MAX_PX}] px", (20, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(out, "FACE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(out, "NO FACE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            status = "REC" if recording else "IDLE"
            npts = len(selected_idxs) if selected_idxs is not None else 0
            cv2.putText(out, f"{status} | speaker: {SPEAKER} | label: {current_label} | pts: {npts}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, "1-0 to set label | r to toggle record | q to quit",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

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
                    print("Recording started")
                    buffer_X, buffer_ts, buffer_roi = [], [], []
                    selected_idxs = None
                    prev_xy_norm = None
                else:
                    if len(buffer_X) > 5 and selected_idxs is not None:
                        X = np.stack(buffer_X).astype(np.float32)
                        ts = np.array(buffer_ts, dtype=np.int32)

                        save_dict = dict(
                            X=X,
                            ts=ts,
                            label=current_label,
                            speaker=SPEAKER,
                            idxs=np.array(selected_idxs, dtype=np.int32),
                        )

                        if SAVE_ROI and len(buffer_roi) > 0:
                            R = np.stack(buffer_roi).astype(np.uint8)
                            T = min(len(X), len(R))
                            save_dict["X"] = X[:T]
                            save_dict["ts"] = ts[:T]
                            save_dict["roi"] = R[:T]

                        fname = f"{SPEAKER}_{current_label}_{int(time.time())}_{clip_id:04d}.npz"
                        np.savez_compressed(os.path.join(OUT_DIR, fname), **save_dict)
                        print(f"saved {fname}")
                        clip_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
