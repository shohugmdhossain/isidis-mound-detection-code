#!/usr/bin/env python3

import os
import csv
import math
from collections import defaultdict

# My deployment folders
MY_FOLDER = "/work/shohug/maskrcnn_new_project/Isidis_deployment_r50_thresholds_nolatlon"
MY_INPUT = os.path.join(MY_FOLDER, "merged")
MY_OUTPUT = os.path.join(MY_FOLDER, "deduped_fast")
os.makedirs(MY_OUTPUT, exist_ok=True)

# Confidence thresholds
MY_THRESHOLDS = ["0p5", "0p6", "0p7", "0p8", "0p9"]

# CTX is about 5 m/pixel, so 400 m is about 80 pixels
MY_PIXEL_SIZE_M = 5.0
MY_DUPLICATE_RADIUS_M = 400.0
MY_DUPLICATE_RADIUS = MY_DUPLICATE_RADIUS_M / MY_PIXEL_SIZE_M

# Use the same size for the search grid
MY_CELL_SIZE = MY_DUPLICATE_RADIUS


def distance(a, b):
    """Pixel distance between two detections in the same mosaic."""
    dx = a["x_global"] - b["x_global"]
    dy = a["y_global"] - b["y_global"]
    return math.hypot(dx, dy)


def cell_of(row):
    """Find the grid cell for this detection."""
    return int(row["x_global"] // MY_CELL_SIZE), int(row["y_global"] // MY_CELL_SIZE)


def is_duplicate(row, cell, grid):
    """Check nearby grid cells for an already kept detection."""
    cx, cy = cell

    for nx in (cx - 1, cx, cx + 1):
        for ny in (cy - 1, cy, cy + 1):
            for kept_row in grid.get((nx, ny), []):
                if distance(row, kept_row) <= MY_DUPLICATE_RADIUS:
                    return True

    return False


# Processing each confidence threshold
for threshold in MY_THRESHOLDS:

    MY_INPUT_FILE = os.path.join(MY_INPUT, f"detections_t{threshold}_merged.csv")
    MY_OUTPUT_FILE = os.path.join(MY_OUTPUT, f"detections_t{threshold}_dedup.csv")

    if not os.path.exists(MY_INPUT_FILE):
        print(f"Missing: {MY_INPUT_FILE}")
        continue

    # Loading detections and calculate their position in the full mosaic
    rows = []

    with open(MY_INPUT_FILE, newline="") as my_file:
        reader = csv.DictReader(my_file)
        fieldnames = reader.fieldnames

        for row in reader:
            row["x_global"] = float(row["tile_x_offset"]) + float(row["xc_pix"])
            row["y_global"] = float(row["tile_y_offset"]) + float(row["yc_pix"])

            row["score_float"] = float(row["score"])
            row["bbox_area_float"] = float(row["bbox_area_pix"])

            rows.append(row)

    # Keep mosaics separate during deduplication
    mosaic_groups = defaultdict(list)

    for row in rows:
        mosaic_groups[row["mosaic_id"]].append(row)

    kept = []

    for mosaic_id, group in mosaic_groups.items():

        # Keep higher-confidence detections first
        group.sort(
            key=lambda x: (x["score_float"], x["bbox_area_float"]),
            reverse=True
        )

        grid = defaultdict(list)

        for row in group:
            cell = cell_of(row)

            if not is_duplicate(row, cell, grid):
                kept.append(row)
                grid[cell].append(row)

    # Sorting the final detections before saving
    kept.sort(
        key=lambda x: (
            x["mosaic_id"],
            float(x["tile_y_offset"]),
            float(x["tile_x_offset"]),
            -x["score_float"]
        )
    )

    # Saving the deduplicated detections
    with open(MY_OUTPUT_FILE, "w", newline="") as my_file:
        writer = csv.DictWriter(my_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in kept:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"{threshold}: input={len(rows)} deduped={len(kept)} removed={len(rows) - len(kept)}")
    print(f"Saved: {MY_OUTPUT_FILE}")

