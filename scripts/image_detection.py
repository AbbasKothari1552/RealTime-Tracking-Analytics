import cv2
from ultralytics import YOLO


# Load YOLOv8 model (automatically downloads if not present)
model = YOLO("yolov8n.pt") 

image = cv2.imread("/workspace/MOT17/train/MOT17-13-FRCNN/img1/000689.jpg")

# Run YOLOv8 inference
results = model(image)  # Class 0 is for 'person'


# Visualize results
annotated_frame = results[0].plot()

cv2.imwrite("/workspace/outputs/result_img_yolo.jpg", annotated_frame)

cv2.destroyAllWindows()