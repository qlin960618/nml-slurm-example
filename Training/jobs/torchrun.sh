#!/bin/bash


source /opt/anaconda3/etc/profile.d/conda.sh
########################
# NOTE: conda environment must be setup on all nodes
########################
conda activate [YOUR_CONDA_ENVIRONMENT]


torchrun --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_GPUS_PER_NODE\
    --master_addr=$1 --master_port=8344 \
    Training/run_torchrun_train.py \
    --epochs=100 \
    --batch_size=12 \
