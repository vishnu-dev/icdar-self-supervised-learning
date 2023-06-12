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

srun python eval.py \
--root-dir=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training \
--label-path=/home/woody/iwfa/iwfa028h/dev/faps/data/ICDAR2017_CLaMM_Training/@ICDAR2017_CLaMM_Training.csv \
--model-name=downstream_linear \
--base-model-name=simclr \
--base-checkpoint=/home/woody/iwfa/iwfa028h/dev/faps/data/trained_models/SimCLR/lightning_logs/version_596745/checkpoints/epoch=372-step=7087.ckpt \
--max-epochs=100 \
--batch-size=64 \
--dataset=icdar
