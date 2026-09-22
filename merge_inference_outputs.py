#!/usr/bin/env python3

import os
import glob

# My deployment folder and merged output folder
MY_FOLDER = "/work/shohug/maskrcnn_new_project/Isidis_deployment_r50_thresholds_nolatlon"
MY_OUTPUT = os.path.join(MY_FOLDER, "merged")
os.makedirs(MY_OUTPUT, exist_ok=True)

# Confidence thresholds
MY_THRESHOLDS = ["0p5", "0p6", "0p7", "0p8", "0p9"]

# Merging the part files for each threshold
for threshold in MY_THRESHOLDS:

    MY_PART_FILES = sorted(glob.glob(os.path.join(MY_FOLDER, f"detections_t{threshold}_part*.csv")))

    if not MY_PART_FILES:
        print(f"No files found for threshold {threshold}")
        continue

    # Final merged file for this threshold
    MY_MERGED_FILE = os.path.join(MY_OUTPUT, f"detections_t{threshold}_merged.csv")
    total_rows = 0

    with open(MY_MERGED_FILE, "w", newline="") as my_output_file:

        # Copying the header from the first file
        with open(MY_PART_FILES[0], "r") as first_file:
            my_output_file.write(first_file.readline())

        # Adding detections from all part files
        for part_file in MY_PART_FILES:
            with open(part_file, "r") as my_file:
                next(my_file)

                for line in my_file:
                    my_output_file.write(line)
                    total_rows += 1

    print(f"Merged {len(MY_PART_FILES)} files -> {MY_MERGED_FILE}")
    print(f"Total detection rows: {total_rows}")
