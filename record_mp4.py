import time
import cv2

# Config
FPS = 30
SECONDS = 5.0
OUT_PATH = "capture.mp4"
    
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video capture")
        
    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUT_PATH,
        fourcc,
        FPS,
        (WIDTH, HEIGHT)
    )

    print("Starting recording in 3 seconds")
    time.sleep(3)
    print("Recording start")
    start_time = time.perf_counter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        out.write(frame)
        cv2.imshow("Recording...", frame)
        
        if time.perf_counter() - start_time >= SECONDS:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()