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

# Defining how areas are going to be splitetd for training, val and test
training_areas = ["Area_1", "Area_2", "Area_3", "Area_4"]
val_areas = ["Area_5"]
test_areas = ["Area_6"]

#------------------------------------------------------------------------------
# ENVIRONMENT AND MODEL PARAMETERS
#------------------------------------------------------------------------------
# Environment (file system and so on) params
eparams = {
    'pc_data_path': r"C:\Users\oliverc\Documents\PG\datasets\Stanford3dDataset_v1.2",
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
# None values are set later, based on environment or load profile
hparams = {
    'epochs': None,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_classes': None,
    'num_workers': None,
    # Max amount of points to sample when classifying objects (not used in segmentation)
    # Mainly used to make all the elements of the classification dataset equal size
    'num_points_per_object': None,
    # Max amount of points to sample per room/space for semantic segmentation
    # (not used in classification)
    # Mainly used when testing semantic segmentation, since
    # to make all the elements of the segmentation dataset equal size when
    # sliding windows are NOT used
    'num_points_per_room': None, 
    # Cols to use from the point cloud files (either 3 (xyz) or 6 (xyzrgb))
    'dimensions_per_object': None,
    # Params to create sliding windows
    'win_width': 1,
    'win_depth': 1,
    'win_height': 3,
    'overlap': 0, # Percentage, 0-95%, 100 will create an infinite loop
}

# Checking if the script is running in GCP
if "OS_IMAGE_FAMILY" in os.environ.keys():
    eparams['pc_data_path'] = "/home/s3disuser/data"

# Some useful info when running with GPUs in pytorch
# torch.cuda.device_count() -> 1 (in our current GCP scenario)
# torch.cuda.get_device_name(0) -> 'Tesla K80' (0 is de device_id from our availbale GPU)
hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Num workers bigger than 0 doesn't work with GPUs (coding parallelization required)
max_workers = 2
hparams['num_workers'] = max_workers if hparams['device'] == 'cpu' else 0

# Select the objects we want to display when visualizing
# table is selected by hand because seems to be the only object detected 
# with 4096 points per room (before deploying sliding windows)
segmentation_target_object = "table"
    
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
                    default = "segmentation",
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
    hparams["num_points_per_room"] = 100
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 3
    

if "low" in args.load:
    hparams["num_points_per_object"] = 100
    hparams["num_points_per_room"] = 512
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 5
   
if "medium" in args.load:
    hparams["num_points_per_object"] = 1024
    hparams["num_points_per_room"] = 1024
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 10
    
if "high" in args.load:
    hparams["num_points_per_object"] = 4096
    hparams["num_points_per_room"] = 4096
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 50
   
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
