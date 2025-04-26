import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize
model = YOLO("/workspace/runs/mot17_yolov8_finetuned4/weights/best.pt")
tracker = DeepSort(
    max_age=30,
    # embedder='mobilenet',
    half=True,
    nn_budget=50
)

# Video setup
cap = cv2.VideoCapture("/workspace/inputs/test.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter("/workspace/outputs/result.mp4", fourcc, fps, (frame_width, frame_height))

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
        verbose=False)

    detections = [
        (box.xywh[0].tolist(), conf.item(), 0)
        for box, conf in zip(results[0].boxes, results[0].boxes.conf)
    ]

    tracks = tracker.update_tracks(detections, frame=frame)
    
    # Visualization
    for track in tracks:
        if not track.is_confirmed(): continue
        box = track.to_ltrb()
        cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                      (int(box[2]), int(box[3])), (0,255,0), 2)
        cv2.putText(frame, f"ID:{track.track_id}", 
                       (int(box[0]), int(box[1]-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # Save frame instead of showing
    out.write(frame)

    frame_count += 1
    if frame_count % 100 == 0:
        elapsed_time = time.time() - start_time
        print(f"FPS: {frame_count / elapsed_time:.2f}")

cap.release()
out.release()

# Benchmark results
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal frames: {frame_count}")
print(f"Processing time: {total_time:.2f}s")
print(f"FPS: {frame_count/total_time:.2f}")

