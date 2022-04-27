#!/bin/bash --login

# Generate required datasets:
# choice:
#     0: KernelPCA
#     1: PCA
#     2: Isomap
python preprocess.py --choice 0
python preprocess.py --choice 1
python preprocess.py --choice 2
