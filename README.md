# clas12-nflows
## 6.862 Applied Machine Learning, Spring 2021, Team 3
Project name: Machine Learning Enhancements to Particle Physics Simulations to Reduce Computational Complexity

Code overview: Sampling target distributions using normalizing flow. Use the [nflow libraries](https://github.com/bayesiains/nflows) that uses pytorch to implement the MAF. The main file is originally from [https://github.com/robertej19/nflows/blob/master/NewMoon.ipynb](https://github.com/robertej19/nflows/blob/master/NewMoon.ipynb).

### Set up an environment

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

### Split data to train/ test

```
cd /nobackup/users/$USER/clas12-nflows
python utils/manual_split.py
```

### Submit the job through batch farm

Following commands will submit one job to slurm farm.
```
cd /nobackup/users/$USER/clas12-nflows
python slurm/submit.py
```
Check logs in slurm/logs/, figures in figures/, models in slurm/models, and generated data in slurm/gendata.

### Test in google colab.

Use the [nflow.ipynb](nflow.ipynb).

(This is not yet updated.)