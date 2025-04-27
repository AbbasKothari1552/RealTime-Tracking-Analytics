from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSORTTracker:
    def __init__(self):
        self.tracker = DeepSort(
                max_age=30,
                embedder='mobilenet',
                half=True,
                nn_budget=50
            )

    def update_tracks(self, detections, frame):
        """
        detections: list of (xmin, ymin, xmax, ymax, confidence, class_id)
        frame: current frame (np.array)
        """
        # formatted_detections = []
        # for det in detections:
        #     bbox = det[:4]  # [xmin, ymin, xmax, ymax]
        #     confidence = det[4]
        #     class_id = det[5]
        #     formatted_detections.append((bbox, confidence, class_id))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks
