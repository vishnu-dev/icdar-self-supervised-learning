#!/bin/bash -l
#SBATCH --job-name=icdar_train_mae
#SBATCH --output=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.out
#SBATCH --error=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=12:00:00

source ~/.profile

conda activate lme

cd ~/dev/pr/icdar-self-supervised-learning/src/

srun python train.py +experiment=mae_fb
