"""
File use to store global vars and required libraries among modules
"""

import os
import argparse
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import logging
import numpy as np
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# If import is not set this way, accessing o3d.ml.torch will throw an error:
# AttributeError: module 'open3d.ml' has no attribute 'torch'

#import open3d.ml.torch as ml3d

# Set the sample path HERE:
# Select what do yo want to visualize: an space or and object
# Uncomment the following line if yoy want to visualize an space
#TEST_PC = "Area_1/office_1/office_1"
# Uncomment the following line if yoy want to visualize an object
TEST_PC = "Area_1/office_1/Annotations/table_1"

BUILDING_DISTRIBUTION = {
    'Building 1': ["Area_1", "Area_3", "Area_6"], 
    'Building 2': ["Area_2", "Area_4"], 
    'Building 3': ["Area_5"], 
}
# Environment (file system and so on) params
eparams = {
    'pc_data_path': "/Users/jgalera/datasets/S3DIS/aligned",
    'pc_file_extension': ".txt",
    'already_rgb_normalized_suffix': "_rgb_norm",
    'pc_file_extensiom_rgb_norm': "_rgb_norm.txt",
    'tensorboard_log_dir': "runs/pointnet_with_s3dis",
    'log_file': "conversion.log",
    's3dis_summary_file': "s3dis_summary.csv",
}

# Model hyperparameters
hparams = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_workers': 0,
    'num_classes': 14,
    'num_points_per_object': 0,
    'dimensions_per_object': 0,
    'epochs': 0,
}

hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parser definition
parser_desc = "Provides convenient out-of-the-box options to train or test "
parser_desc += "a PointNet model based on S3DIS dataset"

parser = argparse.ArgumentParser(prog = "main", 
                    usage = "%(prog)s.py goal=(class|seg) task=(train|test) profile=(low|medium|high)",
                    description = parser_desc)

parser.add_argument("--goal", 
                    "-g",
                    metavar = "goal",
                    type = str,
                    action = "store",
                    nargs = 1,
                    default = "classification",
                    choices = ["class", "seg"],
                    help = "Either classification (class) or segmentation (seg)")

parser.add_argument("--task",
                    "-t", 
                    metavar = "task",
                    type = str,
                    action = "store",
                    nargs = 1,
                    default = "train",
                    choices = ["train", "test"],
                    help = "Either train or test")

parser.add_argument("--load",
                    "-l",
                    metavar = "load",
                    type = str,
                    action = "store",
                    nargs = 1,
                    default = "low",
                    choices = ["low", "medium", "high"],
                    help = "Either low, medium or high")


# Define the logging settings
# Logging is Python-version sensitive
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# For Python < 3.9 (minor version: 9) 
# encoding argument can't be used
if sys.version_info[1] < 9:
    logging.basicConfig(filename = os.path.join(eparams['pc_data_path'], eparams['log_file']),
        level=logging.WARNING,
        format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename = os.path.join(eparams['pc_data_path'], eparams['log_file']),
        encoding='utf-8', 
        level=logging.WARNING,
        format='%(asctime)s %(message)s')
