from subprocess import Popen, PIPE
import sys, time

import getpass
username = getpass.getuser()


#from Axel Schmidt

def file_len(f):
    for i, l in enumerate(f):
        pass
    return i + 1

for i in range(2, 3):
    command= """#!/bin/bash

#SBATCH --job-name=clas12-nflow{0}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node235
#SBATCH --partition=sched_mit_hill
#SBATCH --error=/pool001/{1}/clas12-nflows/slurm/logs/log_{0}.err
#SBATCH --output=/pool001/{1}/clas12-nflows/slurm/logs/log_{0}.out
#SBATCH --mail-user={1}@mit.edu
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate torch-env
#python gen_NF_samples.py
python train_nflow.py
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
    p.communicate(bytes(command, encoding='utf-8'))
