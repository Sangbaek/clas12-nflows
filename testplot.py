import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
# dvpi0p = pd.read_pickle("data/pi0.pkl")

# print(dvpi0p.head(2))
# sys.exit()

# dvpi0p.loc[:, "Gpx"] = dvpi0p.loc[:, "x20"]*np.sin(np.radians(dvpi0p.loc[:, "x21"]))*np.cos(np.radians(dvpi0p.loc[:, "x22"]))
# dvpi0p.loc[:, "Gpy"] = dvpi0p.loc[:, "x20"]*np.sin(np.radians(dvpi0p.loc[:, "x21"]))*np.sin(np.radians(dvpi0p.loc[:, "x22"]))
# dvpi0p.loc[:, "Gpz"] = dvpi0p.loc[:, "x20"]*np.cos(np.radians(dvpi0p.loc[:, "x21"]))
# dvpi0p.loc[:, "Gpx2"] = dvpi0p.loc[:, "x30"]*np.sin(np.radians(dvpi0p.loc[:, "x31"]))*np.cos(np.radians(dvpi0p.loc[:, "x32"]))
# dvpi0p.loc[:, "Gpy2"] = dvpi0p.loc[:, "x30"]*np.sin(np.radians(dvpi0p.loc[:, "x31"]))*np.sin(np.radians(dvpi0p.loc[:, "x32"]))
# dvpi0p.loc[:, "Gpz2"] = dvpi0p.loc[:, "x30"]*np.cos(np.radians(dvpi0p.loc[:, "x31"]))
# gam1 = [dvpi0p['Gpx'], dvpi0p['Gpy'], dvpi0p['Gpz']]
# gam2 = [dvpi0p['Gpx2'], dvpi0p['Gpy2'], dvpi0p['Gpz2']]


#dvpi0p = pd.read_pickle("data/pi0_cartesian_test.pkl")
data_path_16 = "gendata/16features/" #All 16 features


dfs16 = []
    
filenames16 = os.listdir(data_path_16)

for f in filenames16:
    df0 = pd.read_pickle(data_path_16+f)
    dfs16.append(df0)

df_nflow_data_16 = pd.concat(dfs16)
dvpi0p = df_nflow_data_16


dvpi0p.loc[:, "Gpx"] = dvpi0p.loc[:, 9]
dvpi0p.loc[:, "Gpy"] = dvpi0p.loc[:, 10]
dvpi0p.loc[:, "Gpz"] = dvpi0p.loc[:, 11]
dvpi0p.loc[:, "Gpx2"] = dvpi0p.loc[:, 13]
dvpi0p.loc[:, "Gpy2"] = dvpi0p.loc[:, 14]
dvpi0p.loc[:, "Gpz2"] = dvpi0p.loc[:, 15]
gam1 = [dvpi0p['Gpx'], dvpi0p['Gpy'], dvpi0p['Gpz']]
gam2 = [dvpi0p['Gpx2'], dvpi0p['Gpy2'], dvpi0p['Gpz2']]


pi0 = [dvpi0p['Gpx']+dvpi0p['Gpx2'], dvpi0p['Gpy']+dvpi0p['Gpy2'], dvpi0p['Gpz']+dvpi0p['Gpz2']]
def dot(vec1, vec2):
    # dot product of two 3d vectors
    return vec1[0]*vec2[0]+vec1[1]*vec2[1]+vec1[2]*vec2[2]
def mag(vec1):
    # L2 norm of vector
    return np.sqrt(dot(vec1, vec1))
dvpi0p.loc[:, "Mpi0"] = np.sqrt((mag(gam1)+mag(gam2))**2 - mag(pi0)**2)
dvpi0p["Mpi0"].hist(bins = 101)
plt.show()
