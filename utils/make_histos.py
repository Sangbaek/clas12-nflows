import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os, subprocess
import math
import shutil
from icecream import ic
from matplotlib.patches import Rectangle
import pandas as pd

def plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=False,pics_dir="none",plot_title="none",
            filename="ExamplePlot",units=["",""]):
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    # Initalize parameters
    x_name = var_names[0]
    y_name = var_names[1]
    xmin = ranges[0][0]
    xmax =  ranges[0][1]
    num_xbins = ranges[0][2]
    ymin = ranges[1][0]
    ymax =  ranges[1][1]
    num_ybins = ranges[1][2]
    x_bins = np.linspace(xmin, xmax, num_xbins) 
    y_bins = np.linspace(ymin, ymax, num_ybins) 

    # Creating plot
    fig, ax = plt.subplots(figsize =(10, 7)) 
    ax.set_xlabel("{} ({})".format(x_name,units[0]))
    ax.set_ylabel("{} ({})".format(y_name,units[1]))

    plt.hist2d(x_data, y_data, bins =[x_bins, y_bins],
        range=[[xmin,xmax],[ymin,ymax]],norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 

    # Adding color bar 
    if colorbar:
        plt.colorbar()

    #plt.tight_layout()  

    
    #Generate plot title
    if plot_title == "none":
        plot_title = '{} vs {}'.format(x_name,y_name)
    
    plt.title(plot_title) 
        
    
    if saveplot:
        #plot_title.replace("/","")
        new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
        print(new_plot_title)
        if not os.path.exists(pics_dir):
            os.makedirs(pics_dir)
        plt.savefig(pics_dir + new_plot_title+".png")
        plt.close()
        print("Figure {} saved to {}".format(new_plot_title,pics_dir))
    else:
        plt.show()

def plot_1dhist(x_data,vars,ranges="none",second_x=None,
            saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False,
            density=False,annotation=None,xlabel_1="first dataset",xlabel_2="second dataste",
            proton_line=False):
    
    # Initalize parameters
    x_name = vars[0]

    if ranges=="none":
        xmin = 0.99*min(x_data)
        xmax =  1.01*max(x_data)
        num_xbins = 100#int(len(x_data)/)
    else:
        xmin = ranges[0]
        xmax =  ranges[1]
        num_xbins = ranges[2]

    x_bins = np.linspace(xmin, xmax, num_xbins) 

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    fig, ax = plt.subplots(figsize =(10, 7)) 


    y, x = np.histogram(x_data, bins=x_bins)
    x = [(a+x[i+1])/2.0 for i,a in enumerate(x[0:-1])]
    hist = pd.Series(y, x)

    # Plot previously histogrammed data
    #ax = pdf.plot(lw=2, label='PDF', legend=True)
    w = abs(hist.index[1]) - abs(hist.index[0])
    bar_0_10 = ax.bar(hist.index, hist.values, width=w,  align='center',color=first_color, alpha = 1,label=xlabel_1)
    #ax.legend(['PDF', 'Random Samples'])


    if second_x is not None:
        y, x = np.histogram(second_x, bins=x_bins)
        x = [(a+x[i+1])/2.0 for i,a in enumerate(x[0:-1])]
        hist = pd.Series(y, x)

        # Plot previously histogrammed data
        #ax = pdf.plot(lw=2, label='PDF', legend=True)
        w = abs(hist.index[1]) - abs(hist.index[0])
        bar_10_100 = ax.bar(hist.index, hist.values, width=w,  align='center',color='blue', alpha = 0.5)
        #bar_0_10 = ax.bar(np.arange(0,10), np.arange(1,11), color="k")
        #bar_0_10 = ax.hist(range=[xmin,xmax], color=first_color, label=xlabel_1)# cmap = plt.cm.nipy_spectral) 

        # create blank rectangle
        if annotation is not None:
            extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
            ax.legend([bar_0_10, bar_10_100,extra], ("  {}  ".format(xlabel_1),"  {}  ".format(xlabel_2),"EMD Value: {:.4f}".format(annotation)))  
        else:
             ax.legend([bar_0_10, bar_10_100], ("  {}  ".format(xlabel_1),"  {}  ".format(xlabel_2)))  

    if proton_line:
        plt.axvline(x=proton_line,color = 'r', linestyle = '-',label="Pion Mass",linewidth=3.5)
        ax.legend(loc="best")
    # Creating plot
        
    ax.set_xlabel(x_name)  
    ax.set_ylabel('Counts')  
    

    
    #bar0 = ax.plot([], [], ' ', label="EMD Value: {:.2f}".format(annotation))
    

    #plt.tight_layout()  


    #Generate plot title
    if plot_title == "none":
        plot_title = '{} counts'.format(x_name)
    
    plt.title(plot_title) 
    
    if sci_on:
        plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))

    if saveplot:
        new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
        print(new_plot_title)
        


        plt.savefig(pics_dir + new_plot_title+".png")
        plt.close()
    else:
        plt.show()

def plot_several_histo_1D(real_vals, gen_vals, label_real="Physics Data", label_gen="NFlow Model", col2 = "blue",title="Physics vs NFlow Models", saveloc=None):
    fig, axes = plt.subplots(1, num_features, figsize=(4*5, 5))
    for INDEX, ax in zip((0, 1, 2,3 ), axes):
        _, bins, _ = ax.hist(real_vals[:, INDEX], bins=100, color = "red", label=label_real, density=True)
        ax.hist(gen_vals[:, INDEX], bins=bins, label=label_gen, color = col2,alpha=0.5, density=True)
        ax.legend(loc="lower left")
        ax.set_title("Feature {}".format(INDEX) )
    plt.tight_layout()
    if saveloc is not None: plt.savefig(saveloc)
    # plt.show()

if __name__ == "__main__":
    ranges = [0,1,100,0,300,120]
    variables = ['xB','Phi']
    conditions = "none"
    datafile = "F18In_168_20210129/skims-168.pkl"
    plot_2dhist(datafile,variables,ranges)