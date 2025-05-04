# Fine tuned YOLOv8n model path 
# YOLO_MODEL_PATH = "/workspace/runs/mot17_yolov8_finetuned/weights/best.pt"
YOLO_MODEL_PATH = "/workspace/yolov8n.pt"

VIDEO_PATHS = [
    "/workspace/multi_camera_tracking/videos/cam-1.mp4",
    "/workspace/multi_camera_tracking/videos/cam-2.mp4",
    "/workspace/multi_camera_tracking/videos/cam-3.mp4",
    "/workspace/multi_camera_tracking/videos/cam-4.mp4",
]

CONFIDENCE_THRESHOLD = 0.4

SIMILARITY_THRESHOLD = 0.8  # Threshold for feature matching