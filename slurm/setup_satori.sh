#!/bin/bash

cd ~/.local/lib/python3.7/site-packages/nflows/transforms
rm __init__.py autoregressive.py coupling.py
wget https://raw.githubusercontent.com/bayesiains/nflows/master/nflows/transforms/__init__.py
wget https://raw.githubusercontent.com/bayesiains/nflows/master/nflows/transforms/autoregressive.py
wget https://raw.githubusercontent.com/bayesiains/nflows/master/nflows/transforms/coupling.py

mkdir UMNN
cd UMNN
rm __init__.py MonotonicNormalizer.py
wget https://raw.githubusercontent.com/bayesiains/nflows/master/nflows/transforms/UMNN/MonotonicNormalizer.py
wget https://raw.githubusercontent.com/bayesiains/nflows/master/nflows/transforms/UMNN/__init__.py