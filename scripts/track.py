import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort    


GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Fine tuned YOLOv8n model 
model = YOLO("/workspace/runs/mot17_yolov8_finetuned/weights/best.pt")
# Deepsort Model
tracker = DeepSort(
    max_age=30,
    # embedder='mobilenet',
    half=True,
    nn_budget=50
)

# Video setup
cap = cv2.VideoCapture("/workspace/inputs/MOT17-02-FRCNN-raw.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter("/workspace/outputs/result_finetuned.mp4", fourcc, fps, (frame_width, frame_height))

tracking_results = []

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # print("Frame no:", cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Detection and tracking (keep your existing code)
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.5,
        half=True,
        device=0,
        # stream=True, 
        classes=0, 
        verbose=False)[0]
    
    # print("YOLO Raw (pixels):", results[0].boxes[0].xywh[0].tolist())
    # print("YOLO Raw (pixels):", results[0].boxes[0].xywhn[0].tolist())

    ######################################
    # DETECTION
    ######################################

    detections = []

    # loop over the detections
    for data in results.boxes.data.tolist():
        
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence and class id to the results list
        detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], data[4], class_id])


    ######################################
    # TRACKING
    ######################################

    # update the tracker with the new detections
    tracks = tracker.update_tracks(detections, frame=frame)
    
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed(): continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), (0,0,255), -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
        
        # === NEW ===: Save tracking result for this detection
        width = xmax - xmin
        height = ymax - ymin
        tracking_results.append([
            frame_count + 1,  # Frame number (starting from 1)
            track_id,
            xmin,
            ymin,
            width,
            height,
            1,   # Confidence (dummy)
            -1,  # Class id (not used)
            -1   # Visibility (not used)
        ])
        
    # Save frame instead of showing
    out.write(frame)

    frame_count += 1
    if frame_count % 100 == 0:
        elapsed_time = time.time() - start_time
        print(f"FPS: {frame_count / elapsed_time:.2f}")

cap.release()
out.release()

# Save all tracking results into a pred.txt file
tracking_results = np.array(tracking_results, dtype=float)  # force numeric type
np.savetxt("/workspace/outputs/mot17_02_frcnn.txt", tracking_results, fmt='%d,%d,%d,%d,%d,%d,%.2f,%d,%d')

# Benchmark results
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal frames: {frame_count}")
print(f"Processing time: {total_time:.2f}s")
print(f"FPS: {frame_count/total_time:.2f}")

