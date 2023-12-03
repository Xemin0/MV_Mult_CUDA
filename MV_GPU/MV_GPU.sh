#!/bin/bash

# Job name
#SBATCH -J MV_GPU

# Request a GPU node and 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

# Request Memory
#SBATCH --mem=40G

#SBATCH -t 00:35:00

#SBATCH -e ./Results/job-%J.err
#SBATCH -o ./Results/job-%J.out

# ==== End of SBATCH settings ==== #
# Check GPU info
nvidia-smi

# Load CUDA and gcc
module load cuda/11.2.0 gcc/10.2

# Compile
nvcc -arch sm_75 -c MV_GPU.cu -o MV_GPU.o
nvcc -arch sm_75 -c main.cu -o main.o
g++ -c MyUtils.cpp -o MyUtils.o

# Link everything together
nvcc main.o MV_GPU.o MyUtils.o -o testMV_GPU.o

# Remove interediate files
rm main.o MV_GPU.o MyUtils.o

# Run
./testMV_GPU.o
