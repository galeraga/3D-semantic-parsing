"""
3D parsing and classification based on S3DIS dataset
"""

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# conda install pyg -c pyg
from torch_geometric.datasets import S3DIS
# from PIL import Image


S3DIS_DATA_PATH = "/Users/jgalera/datasets/S3DIS"

# Params from previous labs. TBD
hparams = {
    'batch_size': 64,
    'num_workers': 0,
    'num_classes': 100,
    'learning_rate': 0.001,
    'num_epochs': 1
}
hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Defining the data sets and data loaders
# https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.S3DIS

train_dataset = S3DIS(
    root = S3DIS_DATA_PATH,
    train = True,
    transform = None,
    pre_transform = None,
    pre_filter = None)


test_dataset = S3DIS(
    root = S3DIS_DATA_PATH,
    train = False,
    test_area = 6,
    transform = None,
    pre_transform = None,
    pre_filter = None)


if __name__ == "__main__":
    
    print(train_dataset)
    print(test_dataset)