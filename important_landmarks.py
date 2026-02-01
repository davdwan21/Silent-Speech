import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"
CAM_INDEX = 1  # change to 0 if needed

DOT_RADIUS = 1
DOT_COLOR = (0, 255, 0)  # BGR
WINDOW_NAME = "Lower Face (No Nose) + Cheeks"

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

                cv2.putText(out, "LOWER FACE + CHEEKS (NO NOSE)", (20, 40),
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
