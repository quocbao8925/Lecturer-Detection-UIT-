import cv2
import os
from ultralytics import YOLO
from tkinter.filedialog import askopenfilename
import tkinter as tk

tk.Tk().withdraw()

video = None
flag = 0
if (os.path.exists(r"D:\WORKSPACE\Notebook\yolo_person\testvideo.mp4") & flag):
    video = r"D:\WORKSPACE\Notebook\yolo_person\testvideo.mp4"
else:
    video = askopenfilename(
        title="Choose video (.mp4)",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )
    print("Chosen:", video)

model = YOLO(r"D:\WORKSPACE\Notebook\yolo_person\best2.pt")

cap = cv2.VideoCapture(video)
cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        source=frame,
        conf=0.5,
        verbose=False,
        tracker="bytetrack.yaml"  # use ByteTrack
    )
    annotated = results[0].plot()

    annotated = cv2.resize(annotated, (640, 360))
    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()