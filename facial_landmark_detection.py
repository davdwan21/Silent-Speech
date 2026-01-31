import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

def draw_landmarks_points(frame, result):
    if not result.face_landmarks:
        return frame

    h, w = frame.shape[:2]
    out = frame.copy()

    for face in result.face_landmarks:
        for lm in face:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(out, (x, y), 2, (0, 255, 0), -1)

    return out

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    cv2.namedWindow("Face Landmarks", cv2.WINDOW_NORMAL)

    t0 = time.monotonic()
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts_ms = int((time.monotonic() - t0) * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            result = landmarker.detect_for_video(mp_image, ts_ms)

            out = draw_landmarks_points(frame, result)
            cv2.imshow("Face Landmarks", out)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
