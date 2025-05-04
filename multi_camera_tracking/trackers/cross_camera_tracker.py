import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.reid_feature_extractor import FeatureExtractor

class CrossCameraTracker:
    def __init__(self, similarity_threshold=0.8):
        self.global_tracks = {}  # Store global tracks across cameras
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = FeatureExtractor()

    def extract_reid_features(self, detections, frame):
        """Extract ReID features for each detection in the frame."""
        features = []
        for det in detections:
            bbox = det['bbox']  # This is already [xmin, ymin, xmax, ymax]
            # Convert bbox to [x, y, w, h] format expected by feature extractor
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            feature_vector = self.feature_extractor.extract(frame, [x, y, w, h])
            features.append({
                'bbox': bbox, 
                'feature': feature_vector, 
                'track_id': det.get('track_id'),  # Include track_id if available
                'confidence': 1.0  # Default confidence since not provided in track_infos
            })        
        return features

    def match_features(self, frame1_features, frame2_features):
        """Match features between two frames (across cameras)."""
        matched_tracks = []
        for obj1 in frame1_features:
            best_match = None
            max_similarity = 0
            for obj2 in frame2_features:
                similarity = cosine_similarity([obj1['feature']], [obj2['feature']])[0][0]

                if similarity > self.similarity_threshold:
                    best_match = obj2
                    max_similarity = similarity
            if best_match:
                matched_tracks.append((obj1, best_match))  # Link across cameras
        return matched_tracks

    def update_global_tracks(self, matched_tracks):
        """Update global tracks with matched features across cameras."""
        
        # Initialize the global ID counter starting from the current size of global_tracks
        global_id_counter = len(self.global_tracks)
        
        for obj1, obj2 in matched_tracks:

            # Ensure that each object includes its track_id and global_id
            if 'track_id' not in obj1:
                obj1['track_id'] = global_id_counter  # Assign track_id if not present
            if 'track_id' not in obj2:
                obj2['track_id'] = global_id_counter  # Assign track_id if not present


            # Convert the bounding boxes to tuples to make them hashable
            obj1_bbox = tuple(obj1['bbox'])  # Convert list to tuple
            obj2_bbox = tuple(obj2['bbox'])  # Convert list to tuple

            # Check if obj1 is already in global_tracks by its bounding box (as key)
            if obj1_bbox not in self.global_tracks:
                obj1['global_id'] = global_id_counter
                # Assign a new global ID to obj1 and add it to the global tracks
                self.global_tracks[global_id_counter] = obj1
                global_id_counter += 1  # Increment the global ID counter

            # Check if obj2 is already in global_tracks by its bounding box (as key)
            if obj2_bbox not in self.global_tracks:
                obj2['global_id'] = global_id_counter
                # Assign a new global ID to obj2 and add it to the global tracks
                self.global_tracks[global_id_counter] = obj2
                global_id_counter += 1  # Increment the global ID counter

        return self.global_tracks


    def associate_and_track(self, frame_data):
        """Main method to associate and track objects across multiple cameras."""
        all_matched_tracks = []
        
        # Loop through each camera's data (frame and detections)
        for camera1_idx, data1 in enumerate(frame_data):
            frame1 = data1['frame']  # The frame from camera1
            detections1 = data1['tracks']  # The detections for camera1's frame

            # Extract ReID features for the detected objects in camera1's frame
            frame1_features = self.extract_reid_features(detections1, frame1)

            # Compare with other camera frames (cross-camera association)
            for camera2_idx, data2 in enumerate(frame_data):
                if camera1_idx != camera2_idx:  # Don't compare the same camera with itself
                    frame2 = data2['frame']  # The frame from camera2
                    detections2 = data2['tracks']  # The detections for camera2's frame

                    # Extract ReID features for the detected objects in camera2's frame
                    frame2_features = self.extract_reid_features(detections2, frame2)

                    # Match the features between camera1 and camera2
                    matched_tracks = self.match_features(frame1_features, frame2_features)

                    # Store the matched tracks for cross-camera linking
                    all_matched_tracks.extend(matched_tracks)

        # Update global tracks after matching features between cameras
        self.update_global_tracks(all_matched_tracks)
        return self.global_tracks
    
    def get_global_id_for_track(self, track_id):
        """Retrieve global ID for a track from the global tracks."""
        for obj in self.global_tracks.values():
            # Ensure 'track_id' exists in the object
            if 'track_id' in obj and obj['track_id'] == track_id:
                return obj['global_id']  # return the global ID associated with the track ID
        return None  # Return None if no match is found
