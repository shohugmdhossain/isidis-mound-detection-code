# Isidis Planitia Mound Detection Code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20369662.svg)](https://doi.org/10.5281/zenodo.20369662)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code used for deep learning-based detection and morphometric analysis of mound landforms in Isidis Planitia on Mars.

The workflow uses **Mask R-CNN (Detectron2)** and CTX orbital imagery for model training, basin-scale deployment, post-processing, polygon export, and data preparation.

## Overview

The workflow has three main stages:

1. **Train** a Mask R-CNN model on labeled CTX image tiles.
2. **Deploy** the model across Isidis Planitia and merge the parallel outputs.
3. **Remove duplicate detections** from overlapping image tiles.

Training and deployment were run on the **LONI HPC cluster** using NVIDIA A100 GPUs. The Slurm scripts included in `slurm/` were used for the parallel jobs.

## Repository Contents

### Training
- `train_maskrcnn_r50.py` — trains Mask R-CNN with a ResNet-50 backbone
- `train_maskrcnn_r101.py` — trains Mask R-CNN with a ResNet-101 backbone

### Deployment
- `run_inference_on_isidis_tiles.py` — runs basin-scale inference on CTX image tiles
- `merge_inference_outputs.py` — merges outputs from parallel array jobs
- `deduplicate_detections.py` — removes duplicate detections from overlapping tiles

### Environment
- `environment/environment.yml` — conda environment file
- `environment/setup_environment.sh` — setup script for environment installation

### Slurm Scripts
- `inference_array.slurm` — Slurm script for basin-scale inference

## Installation

Clone the repository and create the environment:

    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>
    bash environment/setup_environment.sh

Or create the environment directly from the YAML file:

    conda env create -f environment/environment.yml
    conda activate moundenv

The environment includes PyTorch, CUDA, Detectron2, OpenCV, pandas, NumPy, and other required packages.

## Usage

Run the scripts in order.

### 1. Train a model
    python train_maskrcnn_r50.py
or

    python train_maskrcnn_r101.py

### 2. Run basin-scale inference
    sbatch slurm/inference_array.slurm

### 3. Merge and deduplicate detections
    python merge_inference_outputs.py
    python deduplicate_detections.py


> Paths, tile directories, and model checkpoint locations are defined inside the scripts and may need to be edited for your own setup.

## Dataset

The associated dataset, deployment outputs, detection tables, and polygon datasets are archived on Zenodo:

https://doi.org/10.5281/zenodo.20369662

## Citation

If you use this code or dataset, please cite:

Hossain, M. S., & Morra, G. (2026). *Mask R-CNN-Based Mound Detection and Morphometric Dataset of Isidis Planitia on Mars*. Zenodo. https://doi.org/10.5281/zenodo.20369662

## License

This repository is released under the MIT License. See `LICENSE` for details.
