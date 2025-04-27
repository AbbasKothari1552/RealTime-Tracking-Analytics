import motmetrics as mm
import numpy as np
import os

# === Paths ===
gt_file = "/workspace/MOT17/train/MOT17-02-FRCNN/gt/gt.txt"     # Path to your ground truth file
pred_file = "/workspace/outputs/mot17_02_frcnn.txt" # Path to your prediction file

# === Load Ground Truth and Predictions ===
def load_mot_file(file_path, is_gt=True):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split(',')
            frame_id = int(fields[0])
            obj_id = int(fields[1])
            bbox = list(map(float, fields[2:6]))  # [x, y, w, h]
            conf = float(fields[6]) if len(fields) > 6 else 1.0

            # For GT, ignore objects with id == 0 or negative (MOT format convention)
            if is_gt and obj_id <= 0:
                continue
            data.append((frame_id, obj_id, bbox))
    return data

gt_data = load_mot_file(gt_file, is_gt=True)
pred_data = load_mot_file(pred_file, is_gt=False)

# === Organize by frame ===
def group_by_frame(data):
    frames = {}
    for frame_id, obj_id, bbox in data:
        if frame_id not in frames:
            frames[frame_id] = []
        frames[frame_id].append((obj_id, bbox))
    return frames

gt_frames = group_by_frame(gt_data)
pred_frames = group_by_frame(pred_data)

# === Initialize MOTAccumulator ===
acc = mm.MOTAccumulator(auto_id=True)

# === Helper for IoU calculation ===
def bbox_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    else:
        return inter_area / union_area

# === Frame-by-frame update ===
all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

for frame_id in all_frames:
    gt_objs = gt_frames.get(frame_id, [])
    pred_objs = pred_frames.get(frame_id, [])

    gt_ids = [obj[0] for obj in gt_objs]
    gt_bboxes = [obj[1] for obj in gt_objs]

    pred_ids = [obj[0] for obj in pred_objs]
    pred_bboxes = [obj[1] for obj in pred_objs]

    # Compute IoU matrix (1 - IoU because motmetrics expects distance)
    if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
        distance_matrix = np.zeros((len(gt_bboxes), len(pred_bboxes)))
        for i, gt_box in enumerate(gt_bboxes):
            for j, pred_box in enumerate(pred_bboxes):
                iou_score = bbox_iou(gt_box, pred_box)
                distance_matrix[i, j] = 1 - iou_score  # distance = 1 - iou
    else:
        distance_matrix = np.empty((len(gt_bboxes), len(pred_bboxes)))

    acc.update(
        gt_ids, 
        pred_ids, 
        distance_matrix
    )

# === Compute metrics ===
mh = mm.metrics.create()
summary = mh.compute(
    acc,
    metrics=['mota', 'motp', 'idf1', 'precision', 'recall', 'num_switches'],
    name='Overall'
)

# === Print results ===
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
