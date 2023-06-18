#!/bin/bash -l
#SBATCH --job-name=icdar_eval_simclr
#SBATCH --output=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.out
#SBATCH --error=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.err
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8

conda activate lme

cd ~/dev/pr/icdar-self-supervised-learning/src/

srun python eval.py +experiment=simclr_eval
