# distributed_dataparallel_training example



## Getting started

SLURM is a workload manager and job scheduler for Linux and Unix-like kernels, used by many of the world's supercomputers 
and computer clusters. It provides three key functions. First, it allocates exclusive and/or non-exclusive access to resources 
(computer nodes) to users for some duration of time so they can perform work. Second, it provides a framework for starting, 
executing, and monitoring work (typically a parallel job) on a set of allocated nodes. Finally, it arbitrates contention 
for resources by managing a queue of pending work. Here will be a brief introduction of how to use SLURM to run 
distributed training on multiple nodes using the PyTorch DistributedDataParallel Framework on the internal 
[nml-slurm-cluster] cluster.

**Disclaimer**: This is not a comprehensive guide on SLURM. For more information, please refer to the [SLURM documentation](https://slurm.schedmd.com/overview.html). Additionally, this guide is written for the internal [nml-slurm-cluster] cluster. If you are using a different cluster (ex. UTokyo Wisteria/BDEC-01), please refer to the documentation of the specific cluster.


## System Setup
To begin using the system you will have few initial steps to complete from the help of admin as these steps will include 
access to admin permissions to the targeted node
- setup your user account
- setup your user home directory with link to user_data directory (this is subject to change in future shared storage setup)
- setup your conda environment (alternatively you can use virtualenv stored in your storage/data directory)

### 1. Initial login to the cluster
```bash
ssh <username>@10.198.113.104 -p 1023
```


### 2. Setup your user home directory
The storage of your code and data will locate in the following location
- `~/`: This is your home directory which will also contain `<username>_data` linked to the /storage
- `/storage/user_data/<username>_data`: This is where you should store your Dataset and code

Ex. cloning your code from git (you should change the repository link to your own forked repository)
the code here is only an example and **will not run as it is**
```bash
cd user_data
git clone https://gitlab.com/NML/lab-admin/distributed_dataparallel_training-example.git
```

Ex. moving your dataset to the storage (SPARROW is mounted as read-only mode here, so you will ot have the ability to 
modify file on it here)
```bash
cd /mnt/sparrow/...
cp -r <dataset> /storage/user_data/<username>_data/<dataset>
```

### 3. submit job to the cluster
```bash
cd distributed_dataparallel_training-example/Training/jobs
sbatch train_job.sh
```


## Basic SLURM commands 
- `sbatch`: submit a job script to the cluster
```bash
sbatch <job_script.sh>
```
- `squeue`: show the status of jobs in the queue
- `scancel`: cancel a job
```bash
scancel <job_id>
```
- `scontrol`: show the status of nodes in the cluster


## IMPORTANT NOTES
- do not store your dataset/code in your home directory, as the disk space is limited (currently ~200GB) and will be shared
