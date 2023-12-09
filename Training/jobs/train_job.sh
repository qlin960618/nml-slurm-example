#!/bin/bash
#SBATCH --job-name=My_MultiNode_job

#######################################################################################################
## NOTE: nnodes is the number of nodes you want to use
## NOTE: ntasks-per-node should be one, and let the child script handle the number of GPUs
## NOTE: gpus is the total number of GPUs you want to use
## NOTE: Everything beside --gpus seem to be very specific, modifying them may cause command in the torchrun.sh script
#         to hang at srun grabbing head_node ip. alternatively, can
#######################################################################################################
#SBATCH --exclusive --nodes=2 --ntasks-per-node=1 --gpus=7

#SBATCH --chdir=/home/[USERNAME]/[USERNAME]_data/distributed_dataparallel_training-example
#SBATCH --output=/home/[USERNAME]/[USERNAME]_data/distributed_dataparallel_training-example/out/%x-%j.out


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ips=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --all-ip-addresses)
head_node_ips_array=($head_node_ips)
head_node_ip=${head_node_ips_array[0]}

srun -l --ntasks-per-node=1  Training/jobs/torchrun.sh 10.10.0.101     ##### sometime manula ip is needed, ip discover can be finicky
