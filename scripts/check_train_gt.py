import os
import cv2
import pandas as pd

# ==== CONFIG ====
mot_seq_path = '/workspace/MOT_02'  # Path to MOT17 sequence
img_dir = os.path.join(mot_seq_path, 'img1')
gt_path = os.path.join(mot_seq_path, 'gt', 'gt.txt')
save_output = True
output_dir = '/workspace/outputs/mot17_02_annotated'  # Output directory for saving images

if save_output and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==== LOAD GT ====
# Columns: frame, id, x, y, w, h, conf, class, visibility
cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
gt_df = pd.read_csv(gt_path, header=None, names=cols)

# Only keep ground truth (conf == 1) and class == 1 (person)
# gt_df = gt_df[(gt_df['conf'] == 1) & (gt_df['class'] == 1)]

# ==== VISUALIZE FRAME-BY-FRAME ====
image_files = sorted(os.listdir(img_dir))

for img_file in image_files:
    if not img_file.endswith('.jpg'):
        continue

    frame_id = int(os.path.splitext(img_file)[0])
    img_path = os.path.join(img_dir, img_file)
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Couldn't read: {img_path}")
        continue

    # Get annotations for this frame
    frame_data = gt_df[gt_df['frame'] == frame_id]

    for _, row in frame_data.iterrows():
        x1 = int(row['x'])
        y1 = int(row['y'])
        x2 = int(row['x'] + row['w'])
        y2 = int(row['y'] + row['h'])
        obj_id = int(row['id'])
        vis = float(row['vis'])

        # Choose color based on visibility
        color = (0, 255, 0) if vis > 0.5 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"ID:{obj_id} V:{vis:.1f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # # Show image
    # cv2.imshow("MOT17 Annotation", image)
    # key = cv2.waitKey(30)
    # if key == 27:  # ESC to break
    #     break

    # Save image if required
    if save_output:
        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, image)

cv2.destroyAllWindows()
