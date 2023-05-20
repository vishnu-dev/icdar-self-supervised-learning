#!/bin/bash -l
#SBATCH --job-name=icdar_tune_simclr
#SBATCH --output=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.out
#SBATCH --error=/home/hpc/iwfa/iwfa028h/dev/pr/icdar-self-supervised-learning/scripts/.logs/%x_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00

source ~/.profile

conda activate lme

cd ~/dev/pr/icdar-self-supervised-learning/src/

srun python tuner.py \
--root-dir=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training \
--label-path=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv \
--model-name=simclr \
--num-cpus=8 \
--dataset=icdar
