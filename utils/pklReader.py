#!/usr/bin/env python3

import pandas as pd
import argparse
from icecream import ic

parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
args = parser.parse_args()

df = pd.read_pickle(args.fname)

ic(df)
