import os
import shutil

# Set your base paths
source_dir = 'MOT17/train'  # Original MOT17 train folder
target_images = 'dataset/images/train'
target_labels = 'dataset/labels/train'

# Create destination folders
os.makedirs(target_images, exist_ok=True)
os.makedirs(target_labels, exist_ok=True)

# Traverse each MOT17 sequence
for sequence in os.listdir(source_dir):
    seq_path = os.path.join(source_dir, sequence)
    img_path = os.path.join(seq_path, 'img1')
    label_path = os.path.join(seq_path, 'labels_yolo')

    if not os.path.isdir(img_path) or not os.path.isdir(label_path):
        continue

    print(f"Processing {sequence}...")

    for img_file in sorted(os.listdir(img_path)):
        if img_file.endswith('.jpg'):
            # New image name
            new_name = f"{sequence}_{img_file}"
            shutil.copy(
                os.path.join(img_path, img_file),
                os.path.join(target_images, new_name)
            )

            # Corresponding label
            label_file = img_file.replace('.jpg', '.txt')
            label_src = os.path.join(label_path, label_file)
            label_dst = os.path.join(target_labels, new_name.replace('.jpg', '.txt'))

            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
