from ultralytics import YOLO

from configs import config


class YOLODetector:
    def __init__(self):
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.model.conf = config.CONFIDENCE_THRESHOLD

    def detect(self, frame):
        results = self.model.predict(
                    frame,
                    imgsz=640,
                    conf=0.5,
                    half=True,
                    device=0,
                    # stream=True, 
                    classes=0, 
                    verbose=False)[0]
        
        return results