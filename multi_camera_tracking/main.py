import cv2

from detectors.yolo_detector import YOLODetector
from utils.video_loader import MultiVideoLoader
from trackers.deepsort_tracker import DeepSORTTracker
from trackers.cross_camera_tracker import CrossCameraTracker

def main():
    video_loader = MultiVideoLoader()
    detector = YOLODetector()
    trackers = [DeepSORTTracker() for _ in range(4)] # 4 deepsort instance for 4 camera/videos
    cross_camera_tracker = CrossCameraTracker()


    while True:

        frames = video_loader.read_frames()
        

        # Individual instance for each camera/video
        # trackers = [DeepSORTTracker() for _ in range(len(frames))]

        # If all frames are None (videos finished), break
        if all(frame is None for frame in frames):
            break

        output_frames = []
        frame_data = []  # Store frames along with their detections

        for idx, frame in enumerate(frames):
            if frame is None:
                continue

            # Run YOLO Detection
            results = detector.detect(frame)

            # Format detections for tracker
            detections = []
            for data in results.boxes.data.tolist():
                
                # get the bounding box and the class id
                xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                class_id = int(data[5])
                # add the bounding box (x, y, w, h), confidence and class id to the results list
                detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], data[4], class_id])
            
            # # Store the detections with the frame
            # frame_data.append({'frame': frame, 'detections': detections})

            # update the tracker with the new detections for each videos/camera
            tracks = trackers[idx].update_tracks(detections, frame)

            track_infos = []
            # loop over the tracks
            for track in tracks:
                # if the track is not confirmed, ignore it
                if not track.is_confirmed(): continue

                # get the track id and the bounding box
                track_id = track.track_id
                ltrb = track.to_ltrb()

                xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                    ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
                # print(xmin, ymin, xmax, ymax)
                
                track_infos.append({
                    'bbox': [xmin, ymin, xmax, ymax], 
                    'track_id': track_id
                })
                
                # Get the global ID for the current track (from cross-camera tracker)
                global_id = cross_camera_tracker.get_global_id_for_track(track_id)

                # draw the bounding box and the track id
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                # cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), (0,0,255), -1)
                cv2.putText(frame, str(track_id), (xmax, ymax),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(frame, str(global_id), (xmax-40, ymax),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            frame_data.append({'frame': frame, 'tracks': track_infos})

            cv2.imshow(f'Camera {idx+1}', frame)
        
        # Update global tracks with cross-camera association
        global_tracks = cross_camera_tracker.associate_and_track(frame_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_loader.release_all()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
