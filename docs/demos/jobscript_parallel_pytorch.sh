#!/usr/bin/env bash
#SBATCH -A C3SE-STAFF  # change this
#SBATCH --gpus-per-node=A100:4
#SBATCH --ntasks-per-node=4
#SBATCH -N 2
#SBATCH -t 30

cat $0  # log jobscript

module purge

container="$HOME/containers/pytorch-bundle.sif"  # change this

srun apptainer exec "${container}" python torch_gpu_parallel.py "$@"
