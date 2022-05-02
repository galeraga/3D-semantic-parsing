"""
3D parsing and classification based on the S3DIS dataset included in torch_geometric
"""

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# conda install pyg -c pyg
import torch_geometric as tg

# from torch_geometric.datasets import S3DIS
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
# The dataset in torch_geometric includes two folders:
#   1.- raw/
#           .h5 files
#           all_files.txt (contains the names of all the h5 files (ply_data_all_0.h5)
#           room_files.txt (contains the names of all the rooms in all areas), BUT:
#                        1.- In the original dataset, there were 272 rooms in six Areas
#                        2.- room_files.txt includes more than one line for each room, 
#                           for a total of 23585 lines 
#                        3.- The number of lines per room varies, e.g:
#                               - Area_6_pantry_1: 35 lines
#                               - Area_6_openspace_1: 210 lines
#                               (what do these lines mean? 
#                               Why the datset lenght is per line based?)
#   2.- processed / 
#           Here are stored the files (.pt) when instances of the S3DIS class 
#           are created in order to set new train and test datasets.          
#           This files are created EVERY TIME we change the test_area arg in the 
#           test_dataset
#           There should exist a .pt file for every test_area we've selected. 
#           Since I've changed two times the test_area arg, I have the following files:
#           jgalera@MBPJavi processed % ls -hal
#           total 16637936
#           drwxr-xr-x  8 jgalera  staff   256B May  2 08:47 .
#           drwxr-xr-x  6 jgalera  staff   192B Apr 30 18:30 ..
#           -rw-r--r--  1 jgalera  staff   431B May  2 08:47 pre_filter.pt
#           -rw-r--r--  1 jgalera  staff   431B May  2 08:47 pre_transform.pt
#           -rw-r--r--  1 jgalera  staff   634M May  2 08:47 test_1.pt
#           -rw-r--r--  1 jgalera  staff   566M Apr 30 18:33 test_6.pt
#           -rw-r--r--  1 jgalera  staff   3.3G May  2 08:47 train_1.pt
#           -rw-r--r--  1 jgalera  staff   3.4G Apr 30 18:33 train_6.pt


# Splitting the dataset from room_files.txt
train_dataset = tg.datasets.S3DIS(
    root = S3DIS_DATA_PATH,
    train = True,
    transform = None,
    pre_transform = None,
    pre_filter = None
    )


test_dataset = tg.datasets.S3DIS(
    root = S3DIS_DATA_PATH,
    train = False,
    test_area = 1,
    transform = None,
    pre_transform = None,
    pre_filter = None    
    )


train_dataloader = tg.loader.DataLoader(dataset = train_dataset, 
    batch_size = hparams['batch_size'],
    shuffle = True
    )

test_dataloader = tg.loader.DataLoader(dataset = test_dataset, 
    batch_size = hparams['batch_size'],
    shuffle = False
    )


def inspector_dl(dl):
    """
    Utility to get a deeper knowlegde of dataloaders

    Let's assume batch_size = 64 in all outputs

    DataBatch(x=[262144, 6], y=[262144], pos=[262144, 3], batch=[262144], ptr=[65])
    
    data.x: Node feature matrix with shape [num_nodes, num_node_features]
    data.y: Target to train against (may have arbitrary shape), 
    e.g., node-level targets of shape [num_nodes, *] 
    or graph-level targets of shape [1, *]
    data.pos: Node position matrix with shape [num_nodes, num_dimensions]
    
    ptr is related to the batch size...batch_size +1 
    """

    for i in range(1):
        
        # Get the next element of the dataloader iterator
        # to inspect just a few number os batches
        e = next(iter(dl))
        
        # DataBatch(x=[262144, 6], y=[262144], pos=[262144, 3], batch=[262144], ptr=[65])
        print("Dataloader: ", e)
    
        # ['pos', 'ptr', 'y', 'x', 'batch']
        print("Dataloader keys: ", e.keys)
    
        # torch.Size([262144, 6])
        print("Dataloader x key shape: ", e['x'].shape)

        # 262144
        print("Dataloader num_nodes: ", e.num_nodes)
    
        # 6
        print("Dataloader num_features: ", e.num_node_features)

        # <bound method BaseData.size of DataBatch(x=[262144, 6], y=[262144], pos=[262144, 3], batch=[262144], ptr=[65])>
        print("Dataloader size: ", e.size)


def inspector_ds(ds):  
    """
    Utility to get a deeper knowledge of datasets
    """

    # Output (with test_area = 1) 
    # Train dataset: S3DIS(20291)
    # Test dataset: S3DIS(3687)
    # So total dataset = 20291 + 3687 = 23978
    # The most similar number is the amount of lines in room_filelist (23585)
    print("ds: ", ds)

    # Output (with test_area = 1) 
    # Train dataset: 20291
    # Test dataset: 3687
    print("ds len:", len(ds))

    # Ouput ((with test_area=1))
    # Train dataset  
    #       Data(x=[83111936, 6], y=[83111936], pos=[83111936, 3])
    #       We have around 83M points in the train dataset point cloud with xyzrgb (6)
    # Test dataset  
    #       Data(x=[15101952, 6], y=[15101952], pos=[15101952, 3])
    #       We have around 15M points in the train dataset point cloud with xyzrgb (6)
    # So TOTAL CLOUD POINTS = 83111936 + 15101952 = 98213888
    print("ds.data: ", ds.data)

    # Output: 3 (x, y and pos)
    # Data(x=[83111936, 6], y=[83111936], pos=[83111936, 3])
    # Data(x=[15101952, 6], y=[15101952], pos=[15101952, 3])
    print("len(ds.data): ", len(ds.data))


if __name__ == "__main__":
    
    tg.seed_everything(1)

    for ds in [train_dataset, test_dataset]:
        inspector_ds(ds)
    
    for dl in [train_dataloader, test_dataloader]:
        inspector_dl(dl)

