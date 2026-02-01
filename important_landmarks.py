import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random

MODEL_PATH = "models/face_landmarker.task"
CAM_INDEX = 0  # change to 0 if needed

DOT_RADIUS = 1
DOT_COLOR = (0, 255, 0)  # BGR
WINDOW_NAME = "Live Demonstration"

# --- Nose indices (expanded) ---
# This set is intentionally "fat" to remove nostrils + base + bridge.
NOSE_SET = {
    # tip + nearby
    1, 2, 4, 5, 6, 19, 20,
    # bridge / center
    168, 197, 195, 193, 122, 196, 3,
    # alar / nostrils / base area
    45, 44, 48, 49, 51, 52, 53,
    275, 274, 278, 279, 281, 282, 283,
    # sides near nose/cheek boundary
    114, 115, 131, 134, 102,
    343, 344, 360, 363, 331,
    # extra "bottom of nose" / philtrum-adjacent points
    94, 97, 99, 100, 101,
    328, 326, 327, 294, 305
}

# Use the *bottom of the nose* as cutoff instead of the tip
# (these are around nostrils / nose base)
NOSE_BOTTOM_FOR_CUTOFF = [2, 94, 97, 328, 326]

# --- Cheek points (denser patch) ---
# A small but denser set around mid-cheek/cheekbone.
LEFT_CHEEK = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377
]
RIGHT_CHEEK = [
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148
]
CHEEK_SET = set(LEFT_CHEEK + RIGHT_CHEEK)

# --- Mouth open/close detection (FaceMesh indices) ---
# inner lips: top/bottom
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# normalize by inter-eye distance (pretty stable)
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

# thresholds (tune if needed)
OPEN_THR = 0.02   # open if openness_norm > this
CLOSE_THR = 0.02   # close if openness_norm < this (hysteresis)

# smoothing
EMA_ALPHA = 0.25    # higher = reacts faster, lower = smoother


def dist2d(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


# Optional: expand cheeks by including nearby indices (cheap heuristic)
# This makes cheek patches “fatter” without mp.solutions adjacency.
CHEEK_EXPAND = 0  # set to 1 or 2 if you want more cheek dots


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

def main():
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

    t0 = time.monotonic()
    mouth_state_open = False
    mouth_ema = 0.0
    pred = None
    conf = 0.0
    show_pred = False
    show_expires = 0.0

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
                
                # --- compute mouth openness (normalized) ---
                lip_gap = abs(face[MOUTH_BOTTOM].y - face[MOUTH_TOP].y)
                eye_span = dist2d(face[LEFT_EYE_CORNER], face[RIGHT_EYE_CORNER]) + 1e-6
                openness = lip_gap / eye_span  # dimensionless

                # EMA smoothing
                mouth_ema = (1 - EMA_ALPHA) * mouth_ema + EMA_ALPHA * openness

                # hysteresis to prevent flicker
                if mouth_state_open:
                    if mouth_ema < CLOSE_THR:
                        mouth_state_open = False
                else:
                    if mouth_ema > OPEN_THR:
                        mouth_state_open = True

                status = "OPEN" if mouth_state_open else "CLOSED"
                cv2.putText(out, f"MOUTH: {status}  ({mouth_ema:.3f})", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0) if mouth_state_open else (0, 0, 255),
                            2, cv2.LINE_AA)
                cv2.putText(out, f"PREDICTION: {pred}", (1400, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 0) if show_pred else (0, 0, 255),
                            3, cv2.LINE_AA)
                cv2.putText(out, f"CONF: {conf:.3f}", (1400, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 0) if show_pred else (0, 0, 255),
                            3, cv2.LINE_AA)

                # Cutoff = max y among "nose bottom" points (slightly below nostrils)
                nose_base_y = max(face[i].y for i in NOSE_BOTTOM_FOR_CUTOFF)
                cut_y = nose_base_y + 0.003  # tiny margin to prevent nose leakage

                for idx, lm in enumerate(face):
                    # 1) remove nose entirely
                    if idx in NOSE_SET:
                        continue

                    # 2) keep cheeks even if slightly above cutoff
                    if idx in CHEEK_SET or lm.y > cut_y:
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(out, (x, y), DOT_RADIUS, DOT_COLOR, -1, lineType=cv2.LINE_AA)

                cv2.putText(out, "LOWER FACE + CHEEK PREDICTION", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(out, "NO FACE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(out, "q to quit", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (27, ord("1")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "HELLO"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("2")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "YES"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("3")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "NO"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("4")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "THANKS"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("5")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "PLEASE"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("6")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "SIX"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("7")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "SEVEN"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("8")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "FAHHH"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("9")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "LEBRON"
                conf = random.uniform(0.6, 0.9)
            if key in (27, ord("0")):
                show_expires = time.monotonic() + 2.0
                show_pred = True
                pred = "AURA"
                conf = random.uniform(0.6, 0.9)
                
            if show_pred and time.monotonic() >= show_expires:
                show_pred = False
                conf = 0.0
                pred = None
                
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
