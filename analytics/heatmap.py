import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Configurations ===
model = YOLO("/workspace/runs/mot17_yolov8_finetuned/weights/best.pt")
cap = cv2.VideoCapture("/workspace/inputs/test.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("/workspace/outputs/heatmap.mp4", fourcc, fps, (frame_width, frame_height))

# === Initialization ===
frame_count = 0
start_time = time.time()
heatmap = np.zeros((frame_height, frame_width), dtype=np.uint16)

# === Processing Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.5,
        half=True,
        device=0,
        classes=0,
        verbose=False
    )[0]

    # Count people detected (optional, not used here directly)
    num_people = sum(1 for data in results.boxes.data.tolist() if int(data[5]) == 0)

    # !!! Missing part: Updating heatmap based on current detections
    for data in results.boxes.data.tolist():
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2

        if 0 <= center_x < frame_width and 0 <= center_y < frame_height:
            cv2.circle(heatmap, (center_x, center_y), radius=15, color=(1), thickness=-1)

    # Normalize and colorize the heatmap
    if np.max(heatmap) > 0:
        current_heatmap = np.clip(heatmap / np.max(heatmap) * 255, 0, 255).astype(np.uint8)
        current_colored_heatmap = cv2.applyColorMap(current_heatmap, cv2.COLORMAP_JET)
    else:
        current_colored_heatmap = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Write the heatmap frame
    out.write(current_colored_heatmap)

    frame_count += 1
    if frame_count % 100 == 0:
        elapsed_time = time.time() - start_time
        print(f"FPS: {frame_count / elapsed_time:.2f}")

# === Cleanup ===
cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal frames: {frame_count}")
print(f"Processing time: {total_time:.2f}s")
print(f"FPS: {frame_count/total_time:.2f}")