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
#SBATCH --job-name=clas12-nflow{0}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/nobackup1c/users/{1}/clas12-nflows/slurm/logs/log_{0}.txt
#SBATCH -p sched_any, sched_mit_hill, sched_mit_redwine
#SBATCH -F=/nobackup1c/users/{1}/clas12-nflows/slurm/nodelist
#SBATCH --mail-user={1}@mit.edu
#SBATCH --mail-type=ALL

module purge
module load anaconda3/2020.11
eval "$(conda shell.bash hook)"
conda activate torch-env
python /nobackup1c/users/{1}/clas12-nflows/train_nflow.py > /nobackup1c/users/{1}/clas12-nflows/slurm/logs/out_{0}.txt
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
