#!/usr/bin/env python3

import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

# Initializing the Detectron2 logger so we can see training progress
setup_logger()

# My files
MY_DATASET = "/work/shohug/maskrcnn_new_project/final_mounds_dataset"
MY_OUTPUT = "/work/shohug/maskrcnn_new_project/outputs/maskrcnn_mounds_r50_20k"

# Training and validation data
MY_DATA = {
    "mounds_train": (os.path.join(MY_DATASET, "train/annotations/instances_train.json"), os.path.join(MY_DATASET, "train/images")),
    "mounds_val": (os.path.join(MY_DATASET, "val/annotations/instances_val.json"), os.path.join(MY_DATASET, "val/images")),
}

# Registering datasets in detectron2
for dataset_name, (annotation_file, image_folder) in MY_DATA.items():
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(dataset_name, {}, annotation_file, image_folder)

# Createing a custom trainer class so we can track and save the best model automatically
class MoundTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, distributed=False, output_dir=output_folder)

    def build_hooks(self):
        my_hooks = super().build_hooks()

        # Saving the model with the best validation mask AP50
        my_hooks.insert(
            -1,
            hooks.BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                "segm/AP50",
                mode="max",
                file_prefix="model_best",
            ),
        )

        return my_hooks


# Model configuration
cfg = get_cfg()

# Base our config on a standard ResNet-50 Mask R-CNN model from the model zoo.
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Dataset split settings
cfg.DATASETS.TRAIN = ("mounds_train",)
cfg.DATASETS.TEST = ("mounds_val",)
cfg.DATALOADER.NUM_WORKERS = 4

# COCO-pretrained Mask R-CNN weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # mound

# Training settings
# Training hyper-parameters (batch size, learning rate, and total iterations)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 20000
cfg.SOLVER.STEPS = (12000, 16000)  # Dropping the learning rate down later in training
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.TEST.EVAL_PERIOD = 500 # validating every 500 iterations

# Multi-scale training to help the model learn objects at various sizes
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333

# Customizing anchor sizes tuned for smaller mounds found in our CTX image tiles
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[28], [64], [128], [192], [256]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

# ROI settings
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Output folder
cfg.OUTPUT_DIR = MY_OUTPUT
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Check settings
print("Training dataset:", cfg.DATASETS.TRAIN)
print("Validation dataset:", cfg.DATASETS.TEST)
print("Output folder:", cfg.OUTPUT_DIR)
print("Max iterations:", cfg.SOLVER.MAX_ITER)
print("Evaluation period:", cfg.TEST.EVAL_PERIOD)

# Start a new run using COCO-pretrained weights
trainer = MoundTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
