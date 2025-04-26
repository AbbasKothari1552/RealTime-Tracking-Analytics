import cv2
from ultralytics import YOLO

def detect_and_draw_boxes():
    # Load YOLOv8 model (automatically downloads if not present)
    model = YOLO("/workspace/yolov8n.pt") 
    
    # Open video capture (0 for webcam, or file path)
    cap = cv2.VideoCapture("/workspace/inputs/MOT17-14-FRCNN-raw.mp4")  # Replace with your video path or 0 for webcam


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter("/workspace/outputs/result_14_yolo.mp4", fourcc, fps, (frame_width, frame_height))
    
    print("Total frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 inference
        results = model(frame, classes=0)  # Class 0 is for 'person'
        
        # Visualize results
        annotated_frame = results[0].plot()  # Automatically draws boxes and labels

        # Save frame instead of showing
        out.write(annotated_frame)
        
        # # Display the annotated frame
        # cv2.imshow("Person Detection", annotated_frame)
        
        # # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_draw_boxes()