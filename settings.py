"""
File use to store global vars and required libraries among modules
"""

import os
import pandas as pd
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import logging
import numpy as np
import random
from tqdm import tqdm

# If import is not set this way, accessing o3d.ml.torch will throw an error:
# AttributeError: module 'open3d.ml' has no attribute 'torch'
import open3d.ml.torch as ml3d

# Set the sample path HERE:
PC_DATA_PATH = "/Users/jgalera/datasets/S3DIS/byhand"
# Select what do yo want to visualize: an space or and object
# Uncomment the following line if yoy want to visualize an space
#TEST_PC = "Area_1/office_1/office_1"
# Uncomment the following line if yoy want to visualize an object
TEST_PC = "Area_1/office_1/Annotations/table_1"

PC_FILE_EXTENSION = ".txt"
ALREADY_RGB_NORMALIZED_SUFFIX = "_rgb_norm"
PC_FILE_EXTENSION_RGB_NORM = ALREADY_RGB_NORMALIZED_SUFFIX + PC_FILE_EXTENSION
LOG_FILE = "conversion.log"
S3DIS_SUMMARY_FILE = "s3dis_summary.csv"
# All point cloud objects must have the same number of points
# MAX_OBJ_POINTS = 100
BUILDING_DISTRIBUTION = {
    'Building 1': ["Area_1", "Area_3", "Area_6"], 
    'Building 2': ["Area_2", "Area_4"], 
    'Building 3': ["Area_5"], 
}


eparams = {
    'pc_data_path': "/Users/jgalera/datasets/S3DIS/byhand",
    'pc_file_extension': ".txt",
    'already_rgb_normalized_suffix': "_rgb_norm",
    'pc_file_extensiom_rgb_norm': "_rgb_norm.txt",
    'log_file': "conversion.log",
    's3dis_summary_file': "s3dis_summary.csv",
}

hparams = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_workers': 0,
    'num_classes': 2,
    'num_points_per_object': 1000,
    'dimensions_per_object': 6,
    'epochs': 5,
}

hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO: define num_points depending on arg parser
# hparams['num_points_per_object'] = 4096,
# TODO: define num_classes depending on arg parser
# hparams['num_classes'] = 