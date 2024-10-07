#!/bin/bash
#SBATCH --job-name=My_MultiNode_job

#######################################################################################################
## NOTE: nnodes is the number of nodes you want to use
## NOTE: ntasks-per-node should be one, and let the child script handle the number of GPUs
## NOTE: gpus is the total number of GPUs you want to use
## NOTE: Everything beside --gpus seem to be very specific, modifying them may cause command in the torchrun.sh script
#         to hang at srun grabbing head_node ip. alternatively, can manually set head_node_ip and pass it to torchrun.sh
#######################################################################################################
#SBATCH --exclusive --nodes=2 --ntasks-per-node=1 --gpus=4 --nodelist=nml-slurm-node1,nml-slurm-node3

#SBATCH --chdir=[REPLACE_ME]
#SBATCH --output=[REPLACE_ME]/out/%x-%j.out


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ips=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --all-ip-addresses)
head_node_ips_array=($head_node_ips)
head_node_ip=${head_node_ips_array[0]} ## In theory, correct. but sometime will fail for unknown reason.

srun -l --ntasks-per-node=1  Training/jobs/torchrun.sh 10.10.0.101     ##### sometime manula ip input is needed, ip discover can be finicky
