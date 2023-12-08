#!/bin/bash
#SBATCH --job-name=My_MultiNode_job

#######################################################################################################
## NOTE: nnodes is the number of nodes you want to use
## NOTE: ntasks-per-node should be one, and let the child script handle the number of GPUs
## NOTE: gpus is the total number of GPUs you want to use
#######################################################################################################
#SBATCH --exclusive --nodes=2 --ntasks-per-node=1 --gpus=7

#SBATCH --chdir=/home/[USERNAME]/[USERNAME]_data/distributed_dataparallel_training-example
#SBATCH --output=/home/[USERNAME]/[USERNAME]_data/distributed_dataparallel_training-example/out/%x-%j.out

srun -l --ntasks-per-node=1  Training/jobs/torchrun.sh
