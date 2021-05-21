# clas12-nflows
## 6.862 Applied Machine Learning, Spring 2021, Team 3
Project name: Machine Learning Enhancements to Particle Physics Simulations to Reduce Computational Complexity

Code overview: Sampling target distributions using normalizing flow. Use the [nflow libraries](https://github.com/bayesiains/nflows) that uses pytorch to implement the MAF. The main file is originally from [https://github.com/robertej19/nflows/blob/master/NewMoon.ipynb](https://github.com/robertej19/nflows/blob/master/NewMoon.ipynb).

### Set up an environment at Satori

Follow the [instruction](https://mit-satori.github.io/satori-basics.html) to know more about Satori.

Create an conda environment to use torch in MIT Satori.
```
ssh satori-login-001.mit.edu
module load wmlce
```

Download the codes with related libraries.
```
pip install --user nflows UMNN icecream
cd /nobackup/users/$USER/
git clone https://github.com/6862-2021SP-team3/clas12-nflows.git
cd clas12-nflows
./slurm/setup.sh
```

Download the data
```
cd /nobackup/users/$USER/clas12-nflows/data
wget -O epgg.pkl https://www.dropbox.com/s/nm0cq4nomjdc5co/epgg.pkl?dl=0
wget -O pi0.pkl https://www.dropbox.com/s/0o7tw8c416al4zy/pi0.pkl?dl=0
```

Split data for train and test
```
cd /nobackup/users/$USER/clas12-nflows
python utils/manual_split.py
```

Following commands will submit one job to Satori.
```
cd /nobackup/users/$USER/clas12-nflows
python slurm/submit_satori.py
```
Check if python3 is used for job submission, when an error occurs.
Check logs in slurm/logs/, figures in figures/, models in slurm/models, and generated data in slurm/gendata.

### Set up an environment at Engaging

Unlike the Satori, which has preset conda setup, we should create an conda environment to use torch in Holyoke.
```
ssh eofe7.mit.edu
module load anaconda3/2020.11
conda create --name torch-env pytorch torchvision torchaudio cudatoolkit=10.2 matplotlib tensorboard pandas scikit-learn scipy --channel pytorch
conda init bash
conda activate torch-env
```

Manually download nflow related libs using pypi.
```
pip install --user pickle5 nflows UMNN icecream
./setup_satori.sh
conda config --set auto_activate_base false
exit
```

Download the data files
```
ssh eofe7.mit.edu
cd /pool001/$USER/
git clone https://github.com/6862-2021SP-team3/clas12-nflows.git
cd clas12-nflows/data
wget -O epgg.pkl https://www.dropbox.com/s/t7nkp2jfp2uennm/epgg.pkl?dl=0
wget -O pi0_cartesian.pkl https://www.dropbox.com/s/0nkht1xls2tmdrm/pi0_cartesian.pkl?dl=0
python manual_split.py
```

```
conda activate torch-env
```

Download the codes with related libraries.
```
pip install --user nflows UMNN icecream
cd /pool001/$USER/
git clone https://github.com/6862-2021SP-team3/clas12-nflows.git
cd clas12-nflows
./slurm/setup_engaging.sh
```

Download the data
```
cd /pool001/$USER/clas12-nflows/data
wget -O epgg.pkl https://www.dropbox.com/s/nm0cq4nomjdc5co/epgg.pkl?dl=0
wget -O pi0.pkl https://www.dropbox.com/s/0o7tw8c416al4zy/pi0.pkl?dl=0
```

Split data for train and test
```
cd /pool001/$USER/clas12-nflows
python utils/manual_split.py
```

Following commands will submit one job to Satori.
```
cd /pool001/$USER/clas12-nflows
python slurm/submit_engaging.py
```
Check if python3 is used for job submission, when an error occurs.
Check logs in slurm/logs/, figures in figures/, models in slurm/models, and generated data in slurm/gendata.

### Test in google colab.

Use the [nflow.ipynb](nflow.ipynb).

(This is not yet updated.)