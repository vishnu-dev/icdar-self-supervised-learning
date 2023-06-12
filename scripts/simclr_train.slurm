#!/bin/bash -l
#SBATCH --job-name=icdar_train_simclr
#SBATCH --output=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.out
#SBATCH --error=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8

conda activate lme

cd ~/dev/pr/icdar-self-supervised-learning/src/

srun python cli.py +experiment=simclr_long
