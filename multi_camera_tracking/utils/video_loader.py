import cv2
from configs import config

class MultiVideoLoader:
    def __init__(self):
        self.captures = [cv2.VideoCapture(path) for path in config.VIDEO_PATHS]

    def read_frames(self):
        frames = []
        for cap in self.captures:
            ret, frame = cap.read()
            if not ret:
                frames.append(None)
            else:
                frames.append(frame)
        return frames

    def release_all(self):
        for cap in self.captures:
            cap.release()
