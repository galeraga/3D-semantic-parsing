"""
File use to store global vars and required libraries among modules
"""
#------------------------------------------------------------------------------
# IMPORTS
#------------------------------------------------------------------------------
# General imports
import os
import argparse
import datetime
import sys
import logging
import random
from tqdm import tqdm
import warnings
import itertools

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
# from torchinfo import summary

#------------------------------------------------------------------------------
# S3DIS DATASET SPECIFIC INFORMATION
#------------------------------------------------------------------------------
movable_objects_set = {"board", "bookcase", "chair", "table", "sofa", "clutter"}
structural_objects_set = {"ceiling", "door", "floor", "wall", "beam", "column", "window", "stairs", "clutter"}

building_distribution = {
    'Building 1': ["Area_1", "Area_3", "Area_6"], 
    'Building 2': ["Area_2", "Area_4"], 
    'Building 3': ["Area_5"], 
}

#------------------------------------------------------------------------------
# ENVIRONMENT AND MODEL PARAMETERS
#------------------------------------------------------------------------------
# Environment (file system and so on) params
eparams = {
    'pc_data_path': "/Users/jgalera/datasets/S3DIS/aligned",
    'pc_file_extension': ".txt",
    'pc_file_extension_rgb_norm': "_rgb_norm.txt",
    'pc_file_extension_sem_seg_suffix': "_annotated",
    'already_rgb_normalized_suffix': "_rgb_norm",
    's3dis_summary_file': "s3dis_summary.csv",
    "checkpoints_folder": "checkpoints",
    "tnet_outputs": "tnet_outputs",
    'tensorboard_log_dir': "runs/pointnet_with_s3dis",
    'sliding_windows_folder': "sliding_windows"
}

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
    'win_width': 1,
    'win_depth': 1,
    'win_height': 3,
    'overlap': 0, #percentage, 0-95%, 100 will create an infinite loop
}

# Checking if the script is running in GCP
if "OS_IMAGE_FAMILY" in os.environ.keys():
    eparams['pc_data_path'] = "/home/s3disuser/data"

# Some useful info when running with GPUs in pytorch
# torch.cuda.device_count() -> 1 (in our current GCP scenario)
# torch.cuda.get_device_name(0) -> 'Tesla K80' (0 is de device_id from our availbale GPU)
hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

#------------------------------------------------------------------------------
# AUX FOLDER CREATION
#------------------------------------------------------------------------------
# To store checkpoints
checkpoint_folder = os.path.join(eparams["pc_data_path"], eparams["checkpoints_folder"])
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

# To store tnet ouputs
tnet_outputs_folder = os.path.join(eparams["pc_data_path"], eparams["tnet_outputs"])
if not os.path.exists(tnet_outputs_folder):
    os.makedirs(tnet_outputs_folder)

# To store sliding windows 
path_to_root_sliding_windows_folder = os.path.join(eparams["pc_data_path"], eparams['sliding_windows_folder'])
if not os.path.exists(path_to_root_sliding_windows_folder):
    os.makedirs(path_to_root_sliding_windows_folder)

# The folder will follow this convention: w_X_d_Y_h_Z_o_T
chosen_params = 'w' + str(hparams['win_width']) 
chosen_params += '_d' + str(hparams['win_depth'])
chosen_params += '_h' + str(hparams['win_height']) 
chosen_params += '_o' + str(hparams['overlap']) 

path_to_current_sliding_windows_folder = os.path.join(
                path_to_root_sliding_windows_folder, chosen_params)

if not os.path.exists(path_to_current_sliding_windows_folder):
    os.makedirs(path_to_current_sliding_windows_folder)

#------------------------------------------------------------------------------
# PARSER DEFINITION AND DEFAULT SETTINGS
#------------------------------------------------------------------------------
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

parser.add_argument("--objects",
                    "-o",
                    metavar = "objects",
                    type = str,
                    action = "store",
                    nargs = 1,
                    default = "movable",
                    choices = ["movable", "structural", "all"],
                    help = "Target objects: either movable, structural or all")

# Get parser args to decide what the program has to do
args = parser.parse_args()

# Adjust some hyperparameters based on the desired resource consumption
# In GCP, "Our suggested max number of worker in current system is 2, 
# which is smaller than what this DataLoader is going to create. Please
# be aware that excessive worker creation might get DataLoader running 
# slow or even freeze, lower the worker number to avoid potential 
# slowness/freeze if necessary"

# Toy is for testing code in a very quick way, where
# getting the most of the model is NOT the goal
if "toy" in args.load:
    hparams["num_points_per_object"] = 10
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 3
    hparams["max_points_per_space"] = 10
    hparams["max_points_per_sliding_window"] = 512

if "low" in args.load:
    hparams["num_points_per_object"] = 100
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 5 #5 
    hparams["max_points_per_space"] = 1000
    hparams["max_points_per_sliding_window"] = 512

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

# Select the number of classes to work with
if "movable" in args.objects:
    hparams["num_classes"] = len(movable_objects_set)

if "structural" in args.objects:
    hparams["num_classes"] = len(structural_objects_set)

if "all" in args.objects:
    hparams["num_classes"] = len(structural_objects_set.union(movable_objects_set))

# Set the device to CPU to avoid running out of memory in GCP GPU
# when testing segmentation with a whole space/room
if ("segmentation" in args.goal) and ("test" in args.task) and ("OS_IMAGE_FAMILY" in os.environ.keys()):
    hparams['device'] =  'cpu'
