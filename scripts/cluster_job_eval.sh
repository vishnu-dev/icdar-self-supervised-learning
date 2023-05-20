#!/bin/bash -l
#SBATCH --job-name=icdar_eval_simclr
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
--backbone-model-name=simclr \
--backbone-checkpoint=/home/woody/iwfa/iwfa028h/dev/faps/data/trained_models/SimCLR/lightning_logs/version_584360/checkpoints/epoch=474-step=4275.ckpt \
--max-epochs=100 \
--batch-size=64 \
--dataset=icdar
