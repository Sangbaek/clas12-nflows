# clas12-nflows
## 6.862 Applied Machine Learning, Spring 2021, Team 3
Project name: Machine Learning Enhancements to Particle Physics Simulations to Reduce Computational Complexity

Code overview: Sampling target distributions using normalizing flow. Use the [nflow libraries](https://github.com/bayesiains/nflows) that uses pytorch to implement the MAF. The main file is originally from [https://github.com/robertej19/nflows/blob/master/NewMoon.ipynb](https://github.com/robertej19/nflows/blob/master/NewMoon.ipynb).

## Test in google colab.

Use the [nflow.ipynb](nflow.ipynb).

## deploy the code in a slurm farm
Follow the [instruction](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch#install), or simply follow the commands.

### Set up an environment

Create an conda environment to use torch in Holyoke.
```
ssh eofe7.mit.edu
module load anaconda3/2020.11
conda create --name torch-env pytorch torchvision torchaudio cudatoolkit=10.2 matplotlib tensorboard pandas scikit-learn scipy --channel pytorch
conda init bash
exit
```

Manually download nflow related libs using pypi.
```
ssh eofe7.mit.edu
conda activate torch-env
pip install --user pickle5 nflows UMNN icecream
./setup.sh
```

Download the codes with the data files
```
git clone https://github.com/6862-2021SP-team3/clas12-nflows.git
cd clas12-nflows/data
wget -O epgg.pkl https://www.dropbox.com/s/t7nkp2jfp2uennm/epgg.pkl?dl=0
wget -O pi0_cartesian.pkl https://www.dropbox.com/s/0nkht1xls2tmdrm/pi0_cartesian.pkl?dl=0
python utils/manual_split.py
```

Optional) To make conda deactivate default,
```
conda config --set auto_activate_base false
```

### Submit the job through batch farm

Following commands will submit one job to slurm farm.
```
ssh eofe7.mit.edu
cd /nobackup1c/users/$USER/clas12-nflows
python slurm/submit.py
```
Check logs in slurm/logs/, figures in slurm/figures, models in slurm/models, and generated data in slurm/gendata.