#!/bin/bash -l
#SBATCH --job-name=icdar_train_byol
#SBATCH --output=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.out
#SBATCH --error=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8

source ~/.profile

conda activate lme

cd ~/dev/pr/icdar-self-supervised-learning/src/

srun python cli.py \
--root-dir /home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training \
--label-path /home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv \
--max-epochs 200 \
--batch-size 64 \
--model-name byol \
--model-params learning_rate 0.03 \
--model-params num_classes 12 \
--model-params weight_decay 0.0004 \
--model-params projector_hidden_dim 512 \
--model-params projector_out_dim 128
