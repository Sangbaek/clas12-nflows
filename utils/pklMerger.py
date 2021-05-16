#!/usr/bin/env python3

import pandas as pd
import argparse
from icecream import ic
import os, sys

parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
parser.add_argument("-d","--dirname", help="a directory of pickle files", default="./")
parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
args = parser.parse_args()

files = os.listdir(args.dirname)

print(files)

dfs = []

for f in files:
    df = pd.read_pickle(args.dirname+f)
    dfs.append(df)
    
result = pd.concat(dfs)

ic(result)

result.to_pickle(args.dirname+"merged_"+str(len(files))+".pkl")

#df = pd.read_pickle(args.fname)

#ic(df)
