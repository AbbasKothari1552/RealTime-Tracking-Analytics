import os
import pandas as pd
import cv2
from tqdm import tqdm

def convert_mot17_to_yolo(gt_path, img_dir, label_output_dir):
    os.makedirs(label_output_dir, exist_ok=True)

    # Load MOT17 GT annotations
    columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
    df = pd.read_csv(gt_path, header=None, names=columns)

    # Filter best-quality person annotations
    df = df[(df['conf'] == 1) & (df['class'] == 1) & (df['vis'] > 0.3)]

    # Get image dimensions from first frame
    first_img_path = os.path.join(img_dir, "000001.jpg")
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        raise ValueError("Couldn't read first image to get dimensions.")
    img_height, img_width = first_img.shape[:2]

    # Process each frame
    grouped = df.groupby('frame')
    for frame_id, group in tqdm(grouped, desc="Converting to YOLO"):
        yolo_lines = []

        for _, row in group.iterrows():
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            # Convert to normalized YOLO format (x_center, y_center, w, h)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = os.path.join(label_output_dir, f"{frame_id:06d}.txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_lines))

def process_all_sequences(mot17_root):
    sequences = [seq for seq in os.listdir(mot17_root) if "FRCNN" in seq]
    for seq in tqdm(sequences, desc="Processing Sequences"):
        seq_path = os.path.join(mot17_root, seq)
        img_dir = os.path.join(seq_path, "img1")
        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        label_dir = os.path.join(seq_path, "labels_yolo")

        if not os.path.exists(gt_path):
            print(f"Skipping {seq} (gt.txt not found)")
            continue

        convert_mot17_to_yolo(gt_path, img_dir, label_dir)

if __name__ == "__main__":
    MOT17_TRAIN_PATH = "/workspace/MOT17/train"  # Update this to your path
    process_all_sequences(MOT17_TRAIN_PATH)
    print("✅ YOLO label conversion complete!")
