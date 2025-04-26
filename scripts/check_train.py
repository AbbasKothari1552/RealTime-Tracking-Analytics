import cv2
import os

# ==== CONFIG ====
image_path = '/workspace/dataset/images/train/MOT17-13-FRCNN_000024.jpg'       # Path to the image
label_path = '/workspace/dataset/labels/train/MOT17-13-FRCNN_000024.txt'       # Path to the corresponding YOLO label
class_names = ['person']                          # Add more if needed (COCO, MOT17, etc.)
save_output = True                              # Set True to save the output image
output_path = '/workspace/outputs/check_train_024_final.jpg'                        # Output file path if save_output=True

# ==== LOAD IMAGE ====
image = cv2.imread(image_path)
h, w = image.shape[:2]

# ==== READ YOLO LABEL ====
with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    class_id, cx, cy, bw, bh = map(float, parts)
    class_id = int(class_id)

    # Convert from normalized center format to pixel box format
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    label = class_names[class_id] if class_id < len(class_names) else str(class_id)

    # Draw bounding box and label
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ==== DISPLAY ====
# cv2.imshow("Annotation Preview", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ==== SAVE ====
if save_output:
    cv2.imwrite(output_path, image)
