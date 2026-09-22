#!/usr/bin/env python3

import os
import re
import csv
from glob import glob
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

# My files directory
MY_TILES = "/work/shohug/Data/Isidis/tiles_1024"
MY_MODEL = "/work/shohug/maskrcnn_new_project/outputs/maskrcnn_mounds_r50_20k/model_0000999.pth"
MY_OUTPUT = "/work/shohug/maskrcnn_new_project/Isidis_deployment_r50_thresholds_nolatlon"

# Confidence thresholds
MY_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

os.makedirs(MY_OUTPUT, exist_ok=True)
os.makedirs(os.path.join(MY_OUTPUT, "logs"), exist_ok=True)

# Slurm array information
MY_TASK = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
MY_NUM_TASKS = int(os.environ.get("N_ARRAY_TASKS", "1"))

# Model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = MY_MODEL
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cuda"

# Running at the lowest threshold first
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(MY_THRESHOLDS)

# Same anchor settings used during training
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[28], [64], [128], [192], [256]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

# Loading my trained model
predictor = DefaultPredictor(cfg)

# My CTX tiles
MY_TILE_FILES = sorted(glob(os.path.join(MY_TILES, "*.tif")))

if not MY_TILE_FILES:
    raise RuntimeError(f"No .tif files found in {MY_TILES}")

# Split tiles between Slurm tasks
MY_ASSIGNED_TILES = MY_TILE_FILES[MY_TASK::MY_NUM_TASKS]

print(f"Total tiles found: {len(MY_TILE_FILES)}")
print(f"Array task: {MY_TASK}/{MY_NUM_TASKS - 1}")
print(f"Tiles assigned to this task: {len(MY_ASSIGNED_TILES)}")

# Detection information saved in the CSV
MY_HEADER = [
    "image",
    "mosaic_id",
    "tile_x_offset",
    "tile_y_offset",
    "threshold",
    "score",
    "class_id",
    "x1_pix", "y1_pix", "x2_pix", "y2_pix",
    "xc_pix", "yc_pix",
    "mask_area_pix",
    "bbox_area_pix",
    "tile_width",
    "tile_height",
]

MY_FILES = {}
MY_WRITERS = {}

# One output CSV for each threshold
for threshold in MY_THRESHOLDS:
    threshold_tag = str(threshold).replace(".", "p")
    MY_CSV = os.path.join(MY_OUTPUT, f"detections_t{threshold_tag}_part{MY_TASK:03d}.csv")

    my_file = open(MY_CSV, "w", newline="")
    my_writer = csv.writer(my_file)
    my_writer.writerow(MY_HEADER)

    MY_FILES[threshold] = my_file
    MY_WRITERS[threshold] = my_writer

# Example: E080_N16_Mosaic_x824_y0.tif
MY_TILE_PATTERN = re.compile(r"(E\d+_N\d+)_Mosaic_x(\d+)_y(\d+)\.tif$")

# Going through my assigned tiles
for i, tile_file in enumerate(MY_ASSIGNED_TILES, start=1):

    image = cv2.imread(tile_file, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Could not read: {tile_file}")
        continue

    height, width = image.shape[:2]
    filename = os.path.basename(tile_file)

    match = MY_TILE_PATTERN.search(filename)

    if match:
        mosaic_id = match.group(1)
        tile_x_offset = int(match.group(2))
        tile_y_offset = int(match.group(3))
    else:
        mosaic_id = "UNKNOWN"
        tile_x_offset = -1
        tile_y_offset = -1

    # Running Mask R-CNN on this tile
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.zeros((0, 4))
    scores = instances.scores.numpy() if instances.has("scores") else np.zeros((0,))
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros((0,), dtype=int)

    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()
    else:
        masks = np.zeros((len(scores), height, width), dtype=bool)

    # Processing each detected mound
    for j, score in enumerate(scores):

        score = float(score)
        class_id = int(classes[j])

        x1, y1, x2, y2 = [float(v) for v in boxes[j]]

        # Center point of the mound
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0

        mask_area = int(masks[j].sum()) if len(masks) > j else 0
        bbox_area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

        MY_ROW = [
            filename,
            mosaic_id,
            tile_x_offset,
            tile_y_offset,
            None,
            score,
            class_id,
            x1, y1, x2, y2,
            xc, yc,
            mask_area,
            bbox_area,
            width,
            height,
        ]

        # Saving the detection for each threshold it passes
        for threshold in MY_THRESHOLDS:
            if score >= threshold:
                my_row = MY_ROW.copy()
                my_row[4] = threshold
                MY_WRITERS[threshold].writerow(my_row)

    # Progress update
    if i % 100 == 0 or i == len(MY_ASSIGNED_TILES):
        print(f"[task {MY_TASK}] Processed {i}/{len(MY_ASSIGNED_TILES)}")

# Closing my output files
for my_file in MY_FILES.values():
    my_file.close()

