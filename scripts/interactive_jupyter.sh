#!/bin/bash -l
#SBATCH --job-name=interactive_jupyter
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8

conda activate lme

cd $HOME

srun jupyter notebook --ip 0.0.0.0 --port 9998 --no-browser