import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if count % frame_rate == 0:
            cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
        count += 1
    cap.release()


extract_frames("data/sample.mp4", "data/frames")
#print ("OpenCV version:", cv2.__version__)
