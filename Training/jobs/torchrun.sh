#!/bin/bash

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ips=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --all-ip-addresses)
head_node_ips_array=($head_node_ips)
head_node_ip=${head_node_ips_array[0]}

source /opt/anaconda3/etc/profile.d/conda.sh
########################
# NOTE: conda environment must be setup on all nodes
########################
conda activate [YOUR_CONDA_ENVIRONMENT]


torchrun --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$SLURM_GPUS_ON_NODE\
    --master_addr=$head_node_ip --master_port=8344 \
    Training/run_torchrun_train.py \
    --epochs=100 \
    --batch_size=12 \
