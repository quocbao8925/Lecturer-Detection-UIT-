import cv2
from ultralytics import YOLO

def main():
    # 1. Load model
    model_path = r"D:\WORKSPACE\Notebook\yolo_person\best1.pt"  # nếu last.pt ở cùng folder
    model = YOLO(model_path)

    # 2. Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera, trying to switch between idx1 and idx2.")
        return

    # 3. loop for frame-by-frame capture
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read any frames from camera.")
            break
        
        frame = cv2.flip(frame, 1)  # mirror
        # 4. forward all frames to model
        results = model.track(
            source=frame,
            conf=0.5,
            verbose=False,
            tracker="bytetrack.yaml"  # use ByteTrack
        )
        # 5. get annotated frame
        annotated_frame = results[0].plot()


        # 6. Show frameq
        cv2.imshow("YOLO Skeleton - Webcam", annotated_frame)

        # 7. Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8. Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()