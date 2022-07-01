"""
File use to store global vars and required libraries among modules
"""
# General imports
import os
import argparse
import datetime
import sys
import logging
import random
from tqdm import tqdm
import warnings

# Math and DL imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Visualization imports
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# from torchinfo import summary


building_distribution = {
    'Building 1': ["Area_1", "Area_3", "Area_6"], 
    'Building 2': ["Area_2", "Area_4"], 
    'Building 3': ["Area_5"], 
}

# Environment (file system and so on) params
eparams = {
    'pc_data_path': r"C:\Users\marcc\OneDrive\Escritorio\PROJECTE\S3DIS_ANTIC\Stanford3dDataset_v1.2_Aligned_Version",
    'pc_file_extension': ".txt",
    'pc_file_extension_rgb_norm': "_rgb_norm.txt",
    'pc_file_extension_sem_seg_suffix': "_annotated",
    'already_rgb_normalized_suffix': "_rgb_norm",
    's3dis_summary_file': "s3dis_summary.csv",
    "checkpoints_folder": "checkpoints",
    "tnet_outputs": "tnet_outputs",
    'tensorboard_log_dir': r"C:\Users\marcc\OneDrive\Escritorio\PROJECTE\CODE\3D-semantic-parsing\runs", # runs/pointnet_with_s3dis
}

# Checking if the script is running in GCP
if "OS_IMAGE_FAMILY" in os.environ.keys():
    eparams['pc_data_path'] = "/home/s3disuser/data"

# Model hyperparameters
hparams = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_classes': 14,
    'num_workers': 0,
    'num_points_per_object': 0,
    'max_points_per_space': 0,
    'max_points_per_sliding_window': 0,
    'dimensions_per_object': 0,
    'epochs': 0,
}
# Some useful info when running with GPUs in pytorch
# torch.cuda.device_count() -> 1 (in our current GCP scenario)
# torch.cuda.get_device_name(0) -> 'Tesla K80' (0 is de device_id from our availbale GPU)
hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


# Creating the checkpoint folder
checkpoint_folder = os.path.join(eparams["pc_data_path"], eparams["checkpoints_folder"])

if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

tnet_outputs_folder = os.path.join(eparams["pc_data_path"], eparams["tnet_outputs"])
# Creating the folder to store the tnet outputs for visualization
if not os.path.exists(tnet_outputs_folder):
    os.makedirs(tnet_outputs_folder)

# Parser definition
parser_desc = "Provides convenient out-of-the-box options to train or test "
parser_desc += "a PointNet model based on S3DIS dataset"

parser = argparse.ArgumentParser(prog = "main", 
                    usage = "%(prog)s.py goal=(class|seg) task=(train|test) load=(low|medium|high)",
                    description = parser_desc)

parser.add_argument("--goal", 
                    "-g",
                    metavar = "goal",
                    type = str,
                    action = "store",
                    nargs = 1,
                    default = "classification",
                    choices = ["classification", "segmentation"],
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
                    choices = ["toy", "low", "medium", "high"],
                    help = "Either toy, low, medium or high")

# Get parser args to decide what the program has to do
args = parser.parse_args()

# Adjust some hyperparameters based on the desired resource consumption
# In GCP, "Our suggested max number of worker in current system is 2, 
# which is smaller than what this DataLoader is going to create. Please
# be aware that excessive worker creation might get DataLoader running 
# slow or even freeze, lower the worker number to avoid potential 
# slowness/freeze if necessary.

# Toy is for testing code in a very quick way, where
# getting the most of the model is NOT the goal
if "toy" in args.load:
    hparams["num_points_per_object"] = 10
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 3
    hparams["max_points_per_space"] = 10
    hparams["max_points_per_sliding_window"] = 10

if "low" in args.load:
    hparams["num_points_per_object"] = 100 
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 1 #5 
    hparams["max_points_per_space"] = 1000
    hparams["max_points_per_sliding_window"] = 100

if "medium" in args.load:
    hparams["num_points_per_object"] = 1024
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 10
    hparams["max_points_per_space"] = 2000
    hparams["max_points_per_sliding_window"] = 1024

if "high" in args.load:
    hparams["num_points_per_object"] = 4096
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 50
    hparams["max_points_per_space"] = 4096
    hparams["max_points_per_sliding_window"] = 4096

# Set the device to CPU to avoid running out of memory in GCP GPU
# when testing segmentation with a whole space/room
if ("segmentation" in args.goal) and ("test" in args.task) and ("OS_IMAGE_FAMILY" in os.environ.keys()):
    hparams['device'] =  'cpu'
