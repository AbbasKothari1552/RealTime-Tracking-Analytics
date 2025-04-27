import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ============ Configurations ============
VIDEO_PATH = "/workspace/inputs/test.mp4"  # Set to None for real-time webcam. Otherwise, provide path e.g., "inputs/test.mp4"
YOLO_MODEL_PATH = "/workspace/runs/mot17_yolov8_finetuned/weights/best.pt"
OUTPUT_PATH = "/workspace/outputs/tracking_with_analytics.mp4"
CONFIDENCE_THRESHOLD = 0.5

# ============ Initialization ============
model = YOLO(YOLO_MODEL_PATH)
tracker = DeepSort(max_age=30, half=True, nn_budget=50)

# Choose input source
cap = cv2.VideoCapture(0 if VIDEO_PATH is None else VIDEO_PATH)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) != 0 else 30  # fallback if fps not available

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# ============ Analytics Data ============
unique_ids = set()
entry_frame = dict()   
heatmap = np.zeros((frame_height, frame_width), dtype=np.uint16)

frame_count = 0
start_time = time.time()

# ============ Processing Loop ============
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
        half=True,
        device=0,
        classes=0,
        verbose=False
    )[0]

    # --- Prepare detections for DeepSORT ---
    detections = []
    for data in results.boxes.data.tolist():
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], data[4], class_id])

    # --- Update tracker ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- Draw results and update analytics ---
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id

        # Add new track's entry frame
        if track_id not in entry_frame:
            entry_frame[track_id] = frame_count  # Record when the ID first appeared
        
        # Calculate center of the bounding box
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        # Increment heatmap at that point (clip to avoid out of bounds)
        if 0 <= center_x < frame_width and 0 <= center_y < frame_height:
            heatmap[center_y, center_x] += 1

        ltrb = track.to_ltrb()
        xmin, ymin, xmax, ymax = map(int, ltrb)

        # Add track_id to unique set
        unique_ids.add(track_id)

        dwell_frames = frame_count - entry_frame[track_id]
        dwell_time = dwell_frames / fps  # in seconds


        # Draw bounding box and ID
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 40, ymin), (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{track_id}", (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display footfall count
        cv2.putText(frame, f"Footfall: {len(unique_ids)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        #Draw dwell time on the frame
        cv2.putText(frame, f"{dwell_time:.1f}s", (xmin, ymax + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save frame
    out.write(frame)

    # Show real-time (optional)
    # cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1
    if frame_count % 100 == 0:
        elapsed_time = time.time() - start_time
        print(f"FPS: {frame_count / elapsed_time:.2f}")

# ============ Cleanup ============
cap.release()
out.release()
cv2.destroyAllWindows()

# Normalize heatmap to 0-255
heatmap_normalized = np.clip(heatmap / np.max(heatmap) * 255, 0, 255).astype(np.uint8)

# Apply color map
colored_heatmap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

# Save heatmap image
cv2.imwrite('/workspace/outputs/heatmap_result.jpg', colored_heatmap)
print("Heatmap saved successfully!")


# Final reporting
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal frames: {frame_count}")
print(f"Processing time: {total_time:.2f}s")
print(f"Average FPS: {frame_count/total_time:.2f}")
print(f"Total Footfall (Unique People): {len(unique_ids)}")
