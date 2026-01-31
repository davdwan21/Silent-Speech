import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"
CAM_INDEX = 1

DRAW_POINTS = True
DRAW_OUTLINE = True      # draws a simple polyline loop for upper & lower

# Standard lips landmark indices (commonly used with FaceMesh lips)
# Source lists match the classic FACEMESH_LIPS region indices. :contentReference[oaicite:1]{index=1}
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78]

# Useful anchors
LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14

def mouth_openness(face, w, h) -> float:
    def p(i):
        lm = face[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    open_dist = np.linalg.norm(p(UPPER_INNER) - p(LOWER_INNER))
    width = np.linalg.norm(p(LEFT_CORNER) - p(RIGHT_CORNER)) + 1e-6
    return float(open_dist / width)

def draw_lips_only(frame_bgr, result):
    out = frame_bgr.copy()
    if not result.face_landmarks:
        return out

    face = result.face_landmarks[0]
    h, w = out.shape[:2]

    def px(i):
        lm = face[i]
        return int(lm.x * w), int(lm.y * h)

    # Draw points
    if DRAW_POINTS:
        for i in set(UPPER_LIPS + LOWER_LIPS):
            cv2.circle(out, px(i), 1, (0, 255, 0), -1)

    # # Mouth openness label (optional)
    # score = mouth_openness(face, w, h)
    # cv2.putText(out, f"mouth_open={score:.3f}", (20, 40),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return out

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cv2.namedWindow("Lips Only", cv2.WINDOW_NORMAL)

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
            cv2.imshow("Lips Only", out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
