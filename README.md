# distributed_dataparallel_training example

```bash
nml@nml-slurm-node1:~$ scontrol show nodes
NodeName=nml-slurm-node1 Arch=x86_64 CoresPerSocket=8
   CPUAlloc=16 CPUEfctv=16 CPUTot=16 CPULoad=4.05
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:p6000:2(S:0)
   NodeAddr=nml-slurm-node1 NodeHostName=nml-slurm-node1 Version=23.11.0
   OS=Linux 6.2.0-37-generic #38~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov  2 18:01:13 UTC 2
   RealMemory=128600 AllocMem=128600 FreeMem=11598 Sockets=1 Boards=1
   State=ALLOCATED ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=debug
   BootTime=2023-12-07T21:50:56 SlurmdStartTime=2023-12-14T16:55:41
   LastBusyTime=2023-12-19T09:14:28 ResumeAfterTime=None
   CfgTRES=cpu=16,mem=128600M,billing=16
   AllocTRES=cpu=16,mem=128600M
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a

NodeName=nml-slurm-node2 Arch=x86_64 CoresPerSocket=10
   CPUAlloc=20 CPUEfctv=20 CPUTot=20 CPULoad=8.84
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:3070:5(S:0)
   NodeAddr=nml-slurm-node2 NodeHostName=nml-slurm-node2 Version=23.11.0
   OS=Linux 6.2.0-37-generic #38~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov  2 18:01:13 UTC 2
   RealMemory=31900 AllocMem=31900 FreeMem=5501 Sockets=1 Boards=1
   State=ALLOCATED ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=debug
   BootTime=2023-12-07T21:50:20 SlurmdStartTime=2023-12-14T16:55:41
   LastBusyTime=2023-12-19T09:14:28 ResumeAfterTime=None
   CfgTRES=cpu=20,mem=31900M,billing=20
   AllocTRES=cpu=20,mem=31900M
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a

NodeName=nml-slurm-node3 Arch=x86_64 CoresPerSocket=6
   CPUAlloc=12 CPUEfctv=12 CPUTot=12 CPULoad=4.48
   AvailableFeatures=(null)
   ActiveFeatures=(null)
   Gres=gpu:v100:2(S:0)
   NodeAddr=nml-slurm-node3 NodeHostName=nml-slurm-node3 Version=23.11.0
   OS=Linux 6.2.0-39-generic #40~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov 16 10:53:04 UTC 2
   RealMemory=63980 AllocMem=63980 FreeMem=34724 Sockets=1 Boards=1
   State=ALLOCATED ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
   Partitions=debug
   BootTime=2023-12-14T15:23:04 SlurmdStartTime=2023-12-14T16:55:41
   LastBusyTime=2023-12-19T09:14:28 ResumeAfterTime=None
   CfgTRES=cpu=12,mem=63980M,billing=12
   AllocTRES=cpu=12,mem=63980M
   CapWatts=n/a
   CurrentWatts=0 AveWatts=0
   ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a
```

## Getting started

This tutorial and example is based on example code found here: [ddp tutorial series](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series)

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

### Step 1. Initial login to the cluster
```bash
ssh <username>@10.198.113.104 -p 1023
```
notes:
- port `1023`: head node (nml-slurm-node1)
- port `1024`: compute node 2 (nml-slurm-node2) 
- node3 not currently exposed

once login you will be in the login node, you can enable ssh passwordless login to the cluster by appending your public key
in the `~/.ssh/authorized_keys` file. For this, you can find instruction [here](https://www.ssh.com/academy/ssh/copy-id)


### Step 2. Setup your user home directory
The storage of your code and data will locate in the following location
- `~/`: This is your home directory which will also contain `<username>_data` linked to the /storage
- `/storage/user_data/<username>_data`: This is where you should store your Dataset and code
- in many cases, your user home directory may not be synced across all nodes. However, the reference from `/storage/...`
should be.

Ex. cloning your code from git (you should change the repository link to your own forked repository)
the code here is only an example and **will probably not run as it is**
```bash
cd /storage/user_data/[YOUR DIR]
git clone https://gitlab.com/NML/lab-admin/distributed_dataparallel_training-example.git
```

Ex. moving your dataset to the storage (SPARROW is mounted as read-only mode here, so you will ot have the ability to 
modify file on it here)
```bash
cd /mnt/sparrow/...
cp -r <dataset> /storage/user_data/[YOUR DIR]/<dataset>
```

### Step3. 


### Step 4. submit job to the cluster
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
