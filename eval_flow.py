import pickle5 as pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('pdf')
import itertools
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch import optim
import os
import sys
import pandas as pd
from matplotlib import interactive
from matplotlib.patches import Rectangle

from utils import make_histos
from utils.utilities import meter


sys.path.insert(0,'/mnt/c/Users/rober/Dropbox/Bobby/Linux/classes/GAML/GAMLX/nflows/nflows')
from nflows.transforms.autoregressive import MaskedUMNNAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import DiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


data_path = "gendata/4features/" #Just electorn features
data_path_16 = "gendata/16features/" #All 16 features


dfs16 = []
    
filenames16 = os.listdir(data_path_16)

for f in filenames16:
    df0 = pd.read_pickle(data_path_16+f)
    dfs16.append(df0)

df_nflow_data_16 = pd.concat(dfs16)


physics_cuts = False
gen_all_emd = False
gen_1d_histos = True
gen_emd_comp = False

dfs = []
    
filenames = os.listdir(data_path)

for f in filenames:
    df0 = pd.read_pickle(data_path+f)
    dfs.append(df0)

df_nflow_data = pd.concat(dfs)
nflow_data_len = len(df_nflow_data.index)
print("The Generated dataset has {} events".format(nflow_data_len))
df_test_data_all = pd.read_pickle("data/pi0_cartesian_test.pkl")
df_test_data = df_test_data_all
#df_test_data = df_test_data_all.sample(n=nflow_data_len)

# if len(df_nflow_data) > len(df_test_data):
#     df_nflow_data = df_nflow_data.sample(n=len(df_test_data))
# else:
#     df_test_data = df_test_data.sample(n=len(df_nflow_data))

if physics_cuts:
    df = df_nflow_data_16
    # e = 0
    # df['emass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2

    e = 4
    df['pmass'] = np.sqrt(df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2)

    # e = 8
    # df['g1mass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2

    # e = 12
    # df['g2mass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2

    # e = 0
    # df['protonE'] = df[4]
    # df['Etot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    # e = 1
    # df['pxtot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    # e = 2
    # df['pytot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    # e = 3
    # df['pztot'] = df[e] + df[e+4]+df[e+8]+df[e+12]
    # df['NetE'] = np.sqrt(df['Etot']**2 - df['pxtot']**2 - df['pytot']**2 - df['pztot']**2)

    e = 0
    df['emass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2
    epsilon = 0.01
    df2 = df.query("emass2>(0-{}) and emass2<(0+{})".format(epsilon,epsilon))
    #df2 = df.query("NetE>(4.556-{}) and NetE<(4.556+{})".format(epsilon,epsilon))
    #df2 = df.query("protonE<1.475")#.format(epsilon,epsilon))
    print(df)

    df = df.head(len(df2.index))
    print(df2)
    #sys.exit()

    bin_size = [100,100]
    xvals = df["pmass"]
    x_name = "NF Model Proton Mass"
    output_dir = "./"
    #ranges = "none"
    ranges = [.925,.951,100]

    make_histos.plot_1dhist(xvals,[x_name,],ranges=ranges,second_x=None,
                     saveplot=False,pics_dir=output_dir,plot_title="Reconstructed Proton Mass, NF Model",
                     density=False,proton_line=True,first_color="green",xlabel_1="NF Data")
    
    sys.exit()

    x_data = df[1]
    y_data = df[2]
    var_names = ["E Px","E Py"]
    saveplots = True
    output_dir = "./"
    title = "Electron $P_X$ vs. $P_Y$, NF 4-Feat. Model"
    filename = title
    units = ["GeV","Gev"]
    ranges = [[-1.5,1.5,100],[-1.5,1.5,100]]

    #interactive(True)
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=False,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                                filename=filename.replace(" ",""),units=units)

    x_data = df2[1]
    y_data = df2[2]
    title = "Electron $P_X$ vs. $P_Y$, NF 4-Feat. Model, $M_e^2$ Cut"

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=False,
                            saveplot=saveplots,pics_dir=output_dir,plot_title=title.replace("/",""),
                            filename=filename.replace(" ",""),units=units)
    #interactive(False)
    #plt.show()

    sys.exit()

if gen_all_emd:

    num_vars = len(df_nflow_data.columns)
    num_iters = 10
    nflow_set = []
    test_set = []
    nflow_vals = np.empty(shape=(num_vars,num_iters))
    test_vals = np.empty(shape=(num_vars,num_iters))
    for i in range(num_iters):
        df_test_data_0 = df_test_data_all.sample(n=int(nflow_data_len/10))
        df_test_data_1 = df_test_data_all.sample(n=int(nflow_data_len/10))
        df_nflow_data_0 = df_nflow_data.sample(n=int(nflow_data_len/10))
        for feature_num in range(num_vars):
            emd_nflow, _, _ = meter(df_test_data_0.to_numpy(),df_nflow_data_0.to_numpy(),feature_num)
            emd_test_data, _, _ = meter(df_test_data_0.to_numpy(),df_test_data_1.to_numpy(),feature_num)
            #print("EMD for feature {} is {:.4f} vs. {:.4f} from test data, a {:.3f} ratio".format(feature_num,emd_nflow,emd_test_data,emd_nflow/emd_test_data))
            nflow_vals[feature_num][i] = emd_nflow
            test_vals[feature_num][i] = emd_test_data
        # nflow_set.append(nflow_vals)
        # test_set.append(test_vals)
    #print(nflow_vals)
    nflow_mean = np.mean(nflow_vals,axis=1)
    nflow_std = np.std(nflow_vals,axis=1)
    test_mean = np.mean(test_vals,axis=1)
    test_std = np.std(test_vals,axis=1)
    ratio_2 = np.mean(nflow_vals/test_vals,axis=1)

    ratio_std_2 = np.std(nflow_vals/test_vals,axis=1)

    x = np.arange(len(ratio_2))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    fig, ax = plt.subplots(figsize =(10, 7)) 
    
    ax.set_xlabel("Feature Number")  
    ax.set_ylabel("Earth Mover's Distance Ratio")
    #plt.tick_params(labelsize=24)

    emd_plot = plt.errorbar(x,ratio_2,yerr=ratio_std_2,fmt='o',label="EMD Ratio",linewidth=2)
    ideal_line = plt.axhline(y = 1, color = 'r', linestyle = '-',label="Ideal Case",linewidth=2)
    plt.title("Earth Mover's Distance Ratio, 4-Feature Model")
    #ax.legend([emd_plot, ideal_line])
    plt.legend(loc='upper center')

    plt.xlim([-.5,3.5])
    plt.ylim([0,10])
    plt.show()
    plotname = "nflow_emd_4.png"
    plt.savefig(plotname)
    plt.close()








if gen_emd_comp:

    #df_nflow_data = df_nflow_data.sample(n=len(df_nflow_data_16))
    #df_test_data = df_test_data.sample(n=len(df_nflow_data_16))

    print(int(len(df_nflow_data_16.index)))

    num_vars = len(df_nflow_data.columns)
    num_iters = 20
    nflow_set = []
    test_set = []
    nflow_vals = np.empty(shape=(num_vars,num_iters))
    nflow_vals_16 = np.empty(shape=(num_vars,num_iters))
    test_vals = np.empty(shape=(num_vars,num_iters))
    for i in range(num_iters):
        df_test_data_0 = df_test_data_all.sample(n=int(len(df_nflow_data_16.index)))
        df_test_data_1 = df_test_data_all.sample(n=int(len(df_nflow_data_16.index)))
        df_nflow_data_0 = df_nflow_data.sample(n=int(len(df_nflow_data_16.index)))
        for feature_num in range(num_vars):
            emd_nflow, _, _ = meter(df_test_data_0.to_numpy(),df_nflow_data_0.to_numpy(),feature_num)
            emd_nflow16, _, _ = meter(df_test_data_0.to_numpy(),df_nflow_data_16.to_numpy(),feature_num)
            emd_test_data, _, _ = meter(df_test_data_0.to_numpy(),df_test_data_1.to_numpy(),feature_num)
            #print("EMD for feature {} is {:.4f} vs. {:.4f} from test data, a {:.3f} ratio".format(feature_num,emd_nflow,emd_test_data,emd_nflow/emd_test_data))
            nflow_vals[feature_num][i] = emd_nflow
            nflow_vals_16[feature_num][i] = emd_nflow16
            test_vals[feature_num][i] = emd_test_data
        # nflow_set.append(nflow_vals)
        # test_set.append(test_vals)
    #print(nflow_vals)
    nflow_mean = np.mean(nflow_vals,axis=1)
    nflow_std = np.std(nflow_vals,axis=1)
    test_mean = np.mean(test_vals,axis=1)
    test_std = np.std(test_vals,axis=1)
    ratio_2 = np.mean(nflow_vals/test_vals,axis=1)
    ratio_std_2 = np.std(nflow_vals/test_vals,axis=1)

    nflow_mean_16 = np.mean(nflow_vals_16,axis=1)
    nflow_std_16 = np.std(nflow_vals_16,axis=1)
    ratio_16 = np.mean(nflow_vals_16/test_vals,axis=1)
    ratio_std_16 = np.std(nflow_vals_16/test_vals,axis=1)



    x = np.arange(len(ratio_2))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    fig, ax = plt.subplots(figsize =(10, 7)) 
    
    ax.set_xlabel("Feature Number")  
    ax.set_ylabel("Earth Mover's Distance Ratio")
    #plt.tick_params(labelsize=24)

    emd_plot = plt.errorbar(x,ratio_2,yerr=ratio_std_2,fmt='gx',label="EMD Ratio - 4 Feat. Model",linewidth=2)
    emd_plot_16 = plt.errorbar(x,ratio_16,yerr=ratio_std_16,fmt='o',label="EMD Ratio - 16 Feat. Model",linewidth=2)
    ideal_line = plt.axhline(y = 1, color = 'r', linestyle = '-',label="Ideal Case",linewidth=2)
    plt.title("EMD Ratio, 4 and 16 Feat. Models")
    #ax.legend([emd_plot, ideal_line])
    plt.legend(loc='upper right')

    plt.xlim([-.5,3.5])
    plt.ylim([0,10])
    plt.show()
    plotname = "nflow_emd_4.png"
    plt.savefig(plotname)
    plt.close()













if gen_1d_histos:
    output_dir = "hists_1D/"

    import itertools
    parts = ["Electron","Proton","Photon 1","Photon 2"]
    feat = ["Energy","X-Momentum","Y-Momentum","Z-Momentum"]
    a = parts
    b = feat
    #names = map(''.join, itertools.chain(itertools.product(list1, list2), itertools.product(list2, list1)))
    names = [r for r in itertools.product(a, b)]#: print r[0] + r[1]

    
    df = df_nflow_data
    e = 0
    df['emass2'] = df[e]**2-df[e+1]**2-df[e+2]**2-df[e+3]**2
    epsilon = 2
    df_nflow_data = df.query("emass2>(0-{}) and emass2<(0+{})".format(epsilon,epsilon))

    if len(df_nflow_data) > len(df_test_data):
        df_nflow_data = df_nflow_data.sample(n=len(df_test_data))
    else:
        df_test_data = df_test_data.sample(n=len(df_nflow_data))


    # # Uncomment for plotting all 1D histograms
    # for feature_ind in range(4):
    #         name = names[feature_ind]
    #         x_name = "{} {}".format(name[0],name[1])
    #         print("Creating 1 D Histogram for: {} ".format(name))
    #         emd_nflow, _, _ = meter(df_test_data.to_numpy(),df_nflow_data.to_numpy(),feature_ind)
    #         xvals_1 = df_test_data[feature_ind]
    #         xvals_2 = df_nflow_data[feature_ind]
    #         make_histos.plot_1dhist(xvals_1,[x_name,],ranges="none",second_x=xvals_2,
    #                 saveplot=False,pics_dir=output_dir,plot_title="{}, NF 4-Feature Model".format(x_name),density=True,
    #                 annotation=emd_nflow,xlabel_1="Microphysics Data",xlabel_2="NF Model Sample")
    # sys.exit()

    # output_dir = "hists_2D_4F/"
    saveplots = False
    #saveplots = True

    #interactive(True)


    # #electron x mom, proton x mom
    # x,y = 6,11
    # var_names = ["Proton Y-Momentum","Photon 1 Z-Momentum"]
    # title = "Proton $P_X$ vs. Photon 1 $P_Z$"
    # ranges = [[-.75,.75,100],[1,9,100]]

    #electron x mom, proton x mom
    # x,y = 1,5 
    # var_names = ["Electron X-Momentum","Proton X-Momentum"]
    # title = "Electon $P_X$ vs. Proton $P_X$"
    # ranges = [[-1.5,1.5,100],[-.75,.75,100]]

    #electron x mom, electron y mom
    # x,y = 1,2
    # var_names = ["Electron X-Momentum","Electron Y-Momentum"]
    # title = "Electon $P_X$ vs. Electron $P_Y$"
    # ranges = [[-1.5,1.5,100],[-1.5,1.5,100]]

    # #electron x mom, electron y mom
    # x,y = 1,2
    # var_names = ["Electron X-Momentum","Electron Z-Momentum"]
    # title = "Electon $P_X$ vs. Electron $P_Z$"
    # ranges = [[-1.5,1.5,100],[1,7,100]]


    # e = 0
    # df_nflow_data['emass2'] = df_nflow_data[e]**2-df_nflow_data[e+1]**2-df_nflow_data[e+2]**2-df_nflow_data[e+3]**2
    # epsilon = 0.5
    # df_nflow_data = df_nflow_data.query("emass2>(0-{}) and emass2<(0+{})".format(epsilon,epsilon))


    #electron mass squared, electron energy
    x,y = 'emass2',1
    var_names = ["Electron Mass Squared","Electron Z Mom."]
    title = "Electon Mass Squared vs. Electron Z Mom."
    ranges = [[-1.5,1.5,100],[-1.5,1.5,100]]


    title_phys = title+ ", Physics Data"
    title_nf = title + ", NF 4-Feature Model"
    filename = title
    units = ["GeV","Gev"]
    x_data = df_test_data[y]
    y_data = df_test_data[y]
    x_data_nf = df_nflow_data[x]
    y_data_nf = df_nflow_data[y]


    interactive(True)

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=False,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title_phys.replace("/",""),
                                filename=filename,units=units)

    
    make_histos.plot_2dhist(x_data_nf,y_data_nf,var_names,ranges,colorbar=False,
                                saveplot=saveplots,pics_dir=output_dir,plot_title=title_nf.replace("/",""),
                                filename=filename,units=units)


    interactive(False)
    plt.show()