# record_training_data.py
# Optimized recorder for lip reading training - saves both landmarks AND video

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============ CONFIG ============
MODEL_PATH = "models/face_landmarker.task"
LANDMARKS_DIR = "clips_npz"
VIDEOS_DIR = "videos_labeled"
FPS = 30
RECORD_SECONDS = 2.0  # Duration per clip
COUNTDOWN_SECONDS = 3  # Countdown before recording

# Your 5 words
WORDS = ["hello", "water", "thanks", "please", "apple"]

# Landmark indices for mouth
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310,
              311, 312, 13, 82, 81, 42, 183, 78]
LEFT_CORNER = 61
RIGHT_CORNER = 291
UPPER_INNER = 13
LOWER_INNER = 14
MOUTH_SET = sorted(set(UPPER_LIPS + LOWER_LIPS))

os.makedirs(LANDMARKS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)


def get_clip_count(word):
    """Count existing clips for a word."""
    landmark_count = len([f for f in os.listdir(LANDMARKS_DIR) if f.startswith(word + "_")])
    return landmark_count


def get_next_clip_id(word):
    """Get next clip ID for a word."""
    existing = os.listdir(LANDMARKS_DIR)
    count = 1
    while f"{word}_{count:03d}.npz" in existing:
        count += 1
    return count


def mouth_openness(face, w, h):
    def p(i):
        lm = face[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    open_dist = np.linalg.norm(p(UPPER_INNER) - p(LOWER_INNER))
    width = np.linalg.norm(p(LEFT_CORNER) - p(RIGHT_CORNER)) + 1e-6
    return float(open_dist / width)


def polygon_area_xy(pts_xy):
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_mouth_features(face, w, h, prev_xy_norm=None):
    xy = np.array([[face[i].x * w, face[i].y * h] for i in MOUTH_SET], dtype=np.float32)
    left = np.array([face[LEFT_CORNER].x * w, face[LEFT_CORNER].y * h], dtype=np.float32)
    right = np.array([face[RIGHT_CORNER].x * w, face[RIGHT_CORNER].y * h], dtype=np.float32)
    width = float(np.linalg.norm(left - right) + 1e-6)
    center = xy.mean(axis=0)
    xy_norm = (xy - center) / width

    openness = mouth_openness(face, w, h)
    loop_idx = UPPER_LIPS + LOWER_LIPS[::-1]
    loop_xy = np.array([[face[i].x * w, face[i].y * h] for i in loop_idx], dtype=np.float32)
    loop_xy_norm = (loop_xy - center) / width
    area = polygon_area_xy(loop_xy_norm)

    if prev_xy_norm is None:
        vel = 0.0
    else:
        vel = float(np.mean(np.linalg.norm(xy_norm - prev_xy_norm, axis=1)))

    feat = np.concatenate([xy_norm.reshape(-1), np.array([openness, area, vel], dtype=np.float32)], axis=0)
    return feat, xy_norm


def draw_mouth_outline(frame, face, w, h):
    """Draw mouth outline on frame."""
    def px(i):
        lm = face[i]
        return int(lm.x * w), int(lm.y * h)

    # Draw mouth points
    for i in MOUTH_SET:
        cv2.circle(frame, px(i), 2, (0, 255, 0), -1)

    # Draw mouth box
    xs = [face[i].x * w for i in MOUTH_SET]
    ys = [face[i].y * h for i in MOUTH_SET]
    margin = 20
    x1, x2 = int(min(xs) - margin), int(max(xs) + margin)
    y1, y2 = int(min(ys) - margin), int(max(ys) + margin)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def main():
    print("\n" + "="*50)
    print("  LIP READING TRAINING DATA RECORDER")
    print("="*50)
    print(f"\nWords to record: {', '.join(WORDS)}")
    print(f"Record duration: {RECORD_SECONDS}s per clip")
    print(f"Aim for: 50-100 clips per word\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )

    # State
    current_word_idx = 0
    current_word = WORDS[current_word_idx]
    state = "idle"  # idle, countdown, recording
    countdown_start = 0
    record_start = 0
    frame_buffer = []
    landmark_buffer = []
    prev_xy_norm = None
    video_writer = None

    print("Controls:")
    print("  SPACE  = Start recording")
    print("  1-5    = Select word")
    print("  N      = Next word")
    print("  Q      = Quit")
    print("-" * 50)

    t0 = time.monotonic()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            ts_ms = int((time.monotonic() - t0) * 1000)

            # Detect landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            face_detected = len(result.face_landmarks) > 0

            # Draw mouth if detected
            if face_detected:
                face = result.face_landmarks[0]
                display = draw_mouth_outline(display, face, WIDTH, HEIGHT)

            # State machine
            if state == "idle":
                # Show current word and counts
                counts = {w: get_clip_count(w) for w in WORDS}

                # Header
                cv2.putText(display, f"Current: {current_word.upper()}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

                # Clip counts
                y = 80
                for i, w in enumerate(WORDS):
                    color = (0, 255, 255) if w == current_word else (200, 200, 200)
                    indicator = ">" if w == current_word else " "
                    cv2.putText(display, f"{indicator} {i+1}. {w}: {counts[w]} clips",
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y += 25

                # Instructions
                cv2.putText(display, "Press SPACE to record", (20, HEIGHT - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, "Press 1-5 to select word, Q to quit", (20, HEIGHT - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Face status
                if not face_detected:
                    cv2.putText(display, "NO FACE DETECTED", (WIDTH//2 - 100, HEIGHT//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            elif state == "countdown":
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SECONDS - elapsed

                if remaining <= 0:
                    # Start recording
                    state = "recording"
                    record_start = time.time()
                    frame_buffer = []
                    landmark_buffer = []
                    prev_xy_norm = None

                    # Start video writer
                    clip_id = get_next_clip_id(current_word)
                    video_path = os.path.join(VIDEOS_DIR, f"{current_word}_{clip_id:03d}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))
                    print(f"Recording: {current_word}...")
                else:
                    # Show countdown
                    cv2.putText(display, f"Say: {current_word.upper()}", (WIDTH//2 - 100, HEIGHT//2 - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.putText(display, str(int(remaining) + 1), (WIDTH//2 - 30, HEIGHT//2 + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

            elif state == "recording":
                elapsed = time.time() - record_start
                remaining = RECORD_SECONDS - elapsed

                # Recording indicator
                cv2.circle(display, (WIDTH - 40, 40), 15, (0, 0, 255), -1)
                cv2.putText(display, f"REC {remaining:.1f}s", (WIDTH - 120, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, f"Say: {current_word.upper()}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                # Save frame
                frame_buffer.append(frame.copy())
                if video_writer:
                    video_writer.write(frame)

                # Extract and save landmarks
                if face_detected:
                    face = result.face_landmarks[0]
                    feat, prev_xy_norm = extract_mouth_features(face, WIDTH, HEIGHT, prev_xy_norm)
                    landmark_buffer.append(feat)

                if remaining <= 0:
                    # Stop recording
                    state = "idle"
                    if video_writer:
                        video_writer.release()
                        video_writer = None

                    # Save landmarks
                    if len(landmark_buffer) > 5:
                        clip_id = get_next_clip_id(current_word)
                        X = np.stack(landmark_buffer, axis=0).astype(np.float32)
                        landmark_path = os.path.join(LANDMARKS_DIR, f"{current_word}_{clip_id:03d}.npz")
                        np.savez_compressed(landmark_path, X=X, label=current_word)
                        print(f"  Saved: {current_word}_{clip_id:03d} ({len(landmark_buffer)} frames)")
                    else:
                        print(f"  Too few frames, clip discarded")

            cv2.imshow("Lip Reading Recorder", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and state == "idle" and face_detected:
                state = "countdown"
                countdown_start = time.time()
            elif key == ord('n') and state == "idle":
                current_word_idx = (current_word_idx + 1) % len(WORDS)
                current_word = WORDS[current_word_idx]
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')] and state == "idle":
                idx = key - ord('1')
                if idx < len(WORDS):
                    current_word_idx = idx
                    current_word = WORDS[current_word_idx]

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # Final summary
    print("\n" + "="*50)
    print("RECORDING SUMMARY")
    print("="*50)
    for w in WORDS:
        count = get_clip_count(w)
        status = "OK" if count >= 50 else "Need more"
        print(f"  {w}: {count} clips [{status}]")
    print("="*50)


if __name__ == "__main__":
    main()
