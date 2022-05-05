#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH --mem=32G
#SBATCH -n 1  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-12:00:00   # time in d-hh:mm:ss
#SBATCH -p gpu
#SBATCH --gres=gpu:4     # Request four GPUs
#SBATCH -c 12      # cpus per task
#SBATCH -q wildfire

# Always purge modules to ensure consistent environments
module purge
# Load required modules for job's environment
source activate sg_benchmark
##
python3 main.py
