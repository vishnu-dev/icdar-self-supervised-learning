#!/bin/bash -l
#SBATCH --job-name=icdar_train_mae
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8

conda activate lme

cd ~/dev/pr/icdar-self-supervised-learning/src/

srun python run_model.py \
--root-dir=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training \
--label-path=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv \
--max-epochs=100 \
--batch-size=128 \
--dataset=icdar_lightly \
--model-name=mae

# srun python run_model.py \
# --root-dir=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training \
# --label-path=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv \
# --max-epochs=500 \
# --batch-size=256 \
# --model-name=simclr

# srun python run_model.py \
# --root-dir=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training \
# --label-path=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv \
# --max-epochs=100 \
# --batch-size=64 \
# --model-name=byol
