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

# Math and DL imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, jaccard_score, ConfusionMatrixDisplay


# Visualization imports
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import torchvision.transforms as transforms
# from torchinfo import summary

#------------------------------------------------------------------------------
# S3DIS DATASET SPECIFIC INFORMATION
#------------------------------------------------------------------------------
#movable_objects_set = {"board", "bookcase", "chair", "table", "sofa", "clutter"}
#movable_objects_set = {"board", "bookcase", "chair", "table", "sofa"}
#structural_objects_set = {"ceiling", "door", "floor", "wall", "beam", "column", "window", "stairs", "clutter"}


movable_objects_set = {"board", "bookcase", "chair", "table", "sofa"}
structural_objects_set = {"ceiling", "door", "floor", "wall", "beam", "column", "window", "stairs"}

building_distribution = {
    'Building 1': ["Area_1", "Area_3", "Area_6"], 
    'Building 2': ["Area_2", "Area_4"], 
    'Building 3': ["Area_5"], 
}

# Defining how areas are going to be splitetd for training, val and test
training_areas = ["Area_1", "Area_2", "Area_3", "Area_4"]
val_areas = ["Area_5"]
test_areas = ["Area_6"]

# Select the objects we want to display when visualizing
# table is selected by hand because seems to be the only object detected 
# with 4096 points per room (before deploying sliding windows)
segmentation_target_object = "table"

#------------------------------------------------------------------------------
# ENVIRONMENT AND MODEL PARAMETERS
#------------------------------------------------------------------------------
# Environment (file system and so on) params
eparams = {
    'pc_data_path': r"C:\Users\Lluis\Desktop\Projecte2\Stanford3dDataset",
    'pc_file_extension': ".txt",
    'pc_file_extension_rgb_norm': "_rgb_norm.txt",
    #'pc_file_extension_sem_seg_suffix': "_annotated",
    'pc_file_extension_sem_seg_suffix': "_annotated_clutter_free",
    'already_rgb_normalized_suffix': "_rgb_norm",
    #'s3dis_summary_file': "s3dis_summary.csv",
    # The proper summary file to work with depends on the args.parser
    's3dis_summary_file_all': "s3dis_summary_clutter_free_all.csv",
    's3dis_summary_file_movable': "s3dis_summary_clutter_free_movable.csv",
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
    'win_height': 4,
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
    
#------------------------------------------------------------------------------
# VISUALIZATION PARAMETERS
#------------------------------------------------------------------------------

# GT labels in S3DIS Dataset
fparams = {
    'ceiling': 0,
    'clutter': 1,
    'door': 2,
    'floor': 3,
    'wall': 4,
    'beam': 5,
    'board': 6,
    'bookcase': 7,
    'chair': 8,
    'table': 9,
    'column': 10,
    'sofa': 11,
    'window': 12,
    'stairs': 13
}, 

cparams = {
    'Red': [1,0,0],
    'Lime': [0,1,0],
    'Blue': [0,0,1],
    'Yellow': [1,1,0],
    'Cyan': [0,1,1],
    'Magenta': [1,0,1],
    'Dark_green': [0,0.39,0],
    'Deep_sky_blue': [0,0.75,1],
    'Saddle_brown': [0.54,0.27,0.07],
    'Lemon_chiffon': [1,0.98,0.8],
    'Turquoise': [0.25,0.88,0.81],
    'Gold': [1,0.84,0],
    'Orange': [1,0.65,0],
    'Chocolate': [0.82,0.41,0.12],
    'Peru': [0.8,0.52,0.25],
    'Blue_violet': [0.54,0.17,0.88],
    'Dark_grey': [0.66,0.66,0.66],
    'Grey': [0.5,0.5,0.5],
}

vparams = {    
    'str_object_to_visualize': "chair",
    'num_max_points_from_GT_file': 50000,
    'num_max_points_1_object_model': 50000,    
    'board_color': cparams['Red'],
    'bookcase_color': cparams['Lime'],
    'chair_color': cparams['Blue'],
    'table_color': cparams['Yellow'],
    'sofa_color': cparams['Cyan'],
    'ceiling_color': cparams['Magenta'],
    'clutter_color': cparams['Dark_green'],
    'door_color': cparams['Deep_sky_blue'],
    'floor_color': cparams['Saddle_brown'],
    'wall_color': cparams['Lemon_chiffon'],
    'beam_color': cparams['Turquoise'],
    'column_color': cparams['Gold'],
    'window_color': cparams['Orange'],
    'stairs_color': cparams['Chocolate'],
}


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

#------------------------------------------------------------------------------
# PARSER DEFINITION AND DEFAULT SETTINGS
#------------------------------------------------------------------------------
# Parser definition
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
                    choices = ["train", "validation", "test", "watch"],
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

# Toy is for testing code in a very quick way, where
# getting the most of the model is NOT the goal
if "toy" in args.load:
    hparams["num_points_per_object"] = 128
    hparams["num_points_per_room"] = 256  #100 originally
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 20 #3 originally
    

if "low" in args.load:
    hparams["num_points_per_object"] = 100
    hparams["num_points_per_room"] = 512
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 5
   
if "medium" in args.load:
    hparams["num_points_per_object"] = 1024
    hparams["num_points_per_room"] = 1024
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 20
    
if "high" in args.load:
    hparams["num_points_per_object"] = 2048
    hparams["num_points_per_room"] = 4096
    hparams["dimensions_per_object"] = 3
    hparams["epochs"] = 20
   
# Adapt params depending on the target objects we're going to work
if "movable" in args.objects:
    objects_set = movable_objects_set
    eparams["s3dis_summary_file"] = eparams["s3dis_summary_file_movable"]
    
if "structural" in args.objects:
    hparams["num_classes"] = len(structural_objects_set)

if "all" in args.objects:
    objects_set = structural_objects_set.union(movable_objects_set)
    eparams["s3dis_summary_file"] = eparams["s3dis_summary_file_all"]

hparams["num_classes"] = len(objects_set)

# The annotated file will storge only the type of points we're working with:
# annotated_clutter_free_all: 
#   - data and labels for all the objects (except clutter): wall, door, beam,
# annotated_clutter_free_movable: 
#   - data and labels for only movable objects (except clutter): chair, table
eparams["pc_file_extension_sem_seg_suffix"] = eparams["pc_file_extension_sem_seg_suffix"] + "_" + ''.join(args.objects)

# To store sliding windows 
path_to_root_sliding_windows_folder = os.path.join(eparams["pc_data_path"], 
                                        eparams['sliding_windows_folder'])
if not os.path.exists(path_to_root_sliding_windows_folder):
    os.makedirs(path_to_root_sliding_windows_folder)

# The folder will follow this convention: w_X_d_Y_h_Z_o_T
chosen_params = 'w' + str(hparams['win_width']) 
chosen_params += '_d' + str(hparams['win_depth'])
chosen_params += '_h' + str(hparams['win_height']) 
chosen_params += '_o' + str(hparams['overlap']) 

path_to_current_sliding_windows_folder = os.path.join(
                path_to_root_sliding_windows_folder, 
                ''.join(args.objects),
                chosen_params)

if not os.path.exists(path_to_current_sliding_windows_folder):
    os.makedirs(path_to_current_sliding_windows_folder)

#------------------------------------------------------------------------------
# VISUALZIATION SETTINGS    
#------------------------------------------------------------------------------

# Rooms with more bookcases and boards (chairs-tables seem to be detected easily)
target_room_for_visualization = "Area_6_office_10"