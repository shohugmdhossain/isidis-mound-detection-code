#!/bin/bash

#SBATCH --job-name=isidis_r50_deploy
#SBATCH --output=/work/shohug/maskrcnn_new_project/Isidis_deployment_r50_thresholds_nolatlon/logs/deploy_%A_%a.out
#SBATCH --error=/work/shohug/maskrcnn_new_project/Isidis_deployment_r50_thresholds_nolatlon/logs/deploy_%A_%a.err
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --array=0-199

# Activate my Detectron2 environment
source /usr/local/packages/conda/24.3.0/etc/profile.d/conda.sh
conda activate /work/shohug/envs/moundenv

# I am using 200 Slurm array tasks
export N_ARRAY_TASKS=200

# Go to my project folder
cd /work/shohug/maskrcnn_new_project

# Run the Isidis deployment
python code/run_inference_on_isidis_tiles.py
