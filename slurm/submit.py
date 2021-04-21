from subprocess import Popen, PIPE
import sys, time

import getpass
username = getpass.getuser()


#from Axel Schmidt

def file_len(f):
    for i, l in enumerate(f):
        pass
    return i + 1

for i in range(0, 1):
    command= """#!/bin/bash
#SBATCH --job-name=clas12-nflow{0}      # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                        # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1                    # number of gpus per node
#SBATCH --time=02:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --output=/nobackup1c/users/{1}/clas12-nflows/slurm/logs/log_{0}.txt

module purge
module load anaconda3/2020.11
eval "$(conda shell.bash hook)"
conda activate torch-env
python nflows.py
""".format(str(i), username)
    queue=Popen(args=["squeue","-u",username],stdin=None,stdout=PIPE)
    linecount = file_len(queue.stdout)-1
    print("There are ", linecount, "jobs on the queue.")
        
    # If we have too many things on the queue, then wait a minute
    while (linecount > 499):
        print("There are still", linecount, "jobs on the queue. Waiting...")
        sys.stdout.flush()
        time.sleep(60)
        queue=Popen(args=["squeue","-u",username],stdin=None,stdout=PIPE)
        linecount = file_len(queue.stdout)            

    p=Popen(args=["sbatch"],stdin=PIPE);
    p.communicate(command)
