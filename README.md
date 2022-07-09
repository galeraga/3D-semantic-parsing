# 3D-semantic-parsing
Repo to host the UPC AIDL spring 2022 post-graduate project

## Table of Contents
- [Abstract](#abstract)
- [Main goals](#main-goals)
- [The dataset](#the-dataset)
  * [The custom ground truth file](#the-custom-ground-truth-file)
    + [S3DISDataset4Classification](#s3disdataset4classification)
    + [S3DISDataset4Segmentation](#s3disdataset4segmentation)
  * [Sliding windows](#sliding-windows)
  * [The final folder structure](#the-final-folder-structure)
- [The model](#the-model)
  * [TransformationNet](#transformationnet)
    + [Mathematical introduction](#mathematical-introduction)
    + [Topology of the network](#topology-of-the-network)
    + [Visualization of the Outputs](#visualization-of-the-outputs)
  * [BasePointNet](#basepointnet)
  * [ClassificationPointNet](#classificationpointnet)
  * [SegmentationPointNet](#segmentationpointnet)
  * [The data flow and model size](#the-data-flow-and-model-size)
    + [For classification](#for-classification)
    + [For segmentation](#for-segmentation)
  * [Metrics](#metrics)
    + [For Classification](#for-classification)
      - [$F_1$ Score](#-f-1--score)
      - [Area Under the Curve (AUC)](#area-under-the-curve--auc-)
    + [For Segmentation](#for-segmentation)
      - [IoU Score (Intersection over Union):](#iou-score--intersection-over-union--)
- [Main Conclusions](#main-conclusions)
- [Related Work](#related-work)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Abstract
A point cloud is a type of 3D geometric data structure, based on unordered set of vectors.
Each point is a vector of its (x, y, z) coordinate plus extra feature channels such as color, normal, etc.

PointNet: Respects invariance of input points and no voxel grids are needed

PointNet provides a unified deep learning architecture for the following application recognition tasks, based on point cloud inputs:
1) Shape classification 
2) Shape part segmentation
3) Scene semantic parsing

This project'll be focus on implementing only **object classification** and **scene semantic parsing** (no shape part segementation is carried out). As a dataset, the **S3DIS dataset** is going to be used, where every point includes both its spatial coordinates and its color info (xyzrgb).

## Main goals
The main goal is to implement a PointNet architecture in Pytorch that uses the S3DIS dataset in order to perform object classification and indoor scene semantic segmentation. 

The following considerations will be of particular interest:
- How color impacts object detection and semantic segmentation
- Goal 2
- Goal 3
- Goal 4
- Goal 5

## The dataset
The **3D Semantic Parsing of Large-Scale Indoor Spaces (S3DIS)** dataset is going to be used in order to work with the PointNet architecture. 

The **basic dataset statistics** are:

- It contains point clouds from **3 different buildings**, distributed in **6 areas**:
  - Building 1: Area_1, Area_3, Area_6 
  - Building 2: Area_2, Area_4 
  - Building 3: Area_5 
- There're **11 different room types** (WCs, conference rooms, copy rooms, hallways, offices, pantries, auditoriums, storage rooms, lounges, lobbies and open spaces) for a grand total of  **272 rooms** (or spaces) diitributed among all areas.  
- Every room can have up to **14 different object types**. These objects can be classified as:
  - **movable** (boards, bookcases, chairs, tables and sofas)  
  - **structural** (ceilings, doors, floors, walls, beams, columns, windows and stairs). 
  - **clutter** (if an object doesn't belong to any of the previous catagories)

More **advanced statistics** can be found after a slightly deeper analysis:

|<img width="1040" alt="image" src="https://user-images.githubusercontent.com/76537012/175812289-8b537a96-cb85-425e-a47c-5120a31baf9b.png">|
|---|
| <sub> Object classes dict: {0:ceiling, 1:clutter, 2:door, 3:floor, 4:wall, 5:beam, 6:board, 7:bookcase, 8:chair, 9:table, 10:column, 11:sofa, 12:window, 12:window, 13:stairs } </sub>|
| <sub> Room types dict: {0:WC, 1:conferenceRoom, 2:copyRoom, 3:hallway, 4:office, 5:pantry, 6:auditorium, 7:storage, 8:lounge, 9:lobby, 10:openSpace}</sub>|


The original **folder structure** of the S3DIS dataset is the following one:

```
├── Area_N
│   ├── space_X
│   │   ├── space_x.txt (the non-annotated file with the point cloud for this space. It only contains 6 cols per row: XYZRGB)
│   │   ├── Annotations
│   │   │   ├── object_1.txt (the file with the point cloud for object_1 that can be found in Space_X. It contains 6 cols per row: XYZRGB)
|   |   |   ├── ...
│   │   │   ├── object_Y.txt (the file with the point cloud for object_Y that can be found in Space_X. It contains 6 cols per row: XYZRGB)
```  

- `object_Y.txt` is a file containing the point cloud of this particular object that belongs to Space_X (e.g., objects *chair_1.txt, chair_2.txt* or *table_1.txt* from an space/room called *office_1*). This point cloud file has 6 columns (non-normalized XYXRGB values).
- `space_x.txt` is a non-annotated point cloud file containing the sum of of all the point cloud object files (object_1, object_2, ...) located within the `Annotations` folder (e.g., the space file called *Area_1\office_1\office_1.txt* contains the sum of the object files *chair_1.txt, chair_2.txt* and the rest of all the object files located inside the `Annotations` directory). As a consequence, the space/room point cloud file has only 6 columns too (non-normalized XYZRGB values).

Comprehensive information about the original S3DIS dataset can be found at: http://buildingparser.stanford.edu/dataset.html 

From this original S3DIS dataset:

- A custom ground truth file (called s3dis_summary.csv) has been created to speed up the process to get to the point cloud files, avoiding recurrent operating system folder traversals.  
- Two custom datasets have been created to feed the dataloaders, depending on the desired goal (S3DISDataset4Classification and S3DISDataset4Segmentation). 

### The custom ground truth file

A CSV file is generated to host the following information:

| Area          | Space         | Space Label  | Space ID | Object    | Object Points | Object Label | Object ID | Health Status 
|:-------------:|:-------------:|:------------:|:--------:|:---------:|:-------------:|:------------:|:---------:|:------------:|
| Area_N        | Space_X       | Space        | [0-10]   | Object Y  |  Integer      | Object       | [0-13]    |    Good/Bad

Columns meaning:
- Area: Any of the available 6 areas (Area_1, Area_2, Area_3, Area_4, Area_5, Area_6). 
- Space: Any of the available 272 spaces/rooms (office_1, hallway_2,...).
- Space Label: Human-readable text string/label to identify any of the 11 different room types (WCs, conference rooms, copy rooms, hallways, offices, pantries, auditoriums, storage rooms, lounges, lobbies and open spaces).
- Space ID: A dict-mapping for any of the 11 space labels.
- Object: Any of the available 9832 objects distributed among all the different rooms (board_1, bookcase_1, chair_1, chair_2, table_1,...).
- Object Points: Number of points/rows this particular object point cloud file has.
- Object Label: Human-readable text string/label to identify any of the 14 different object types (boards, bookcases, chairs, tables, sofas, ceilings, doors, floors, walls, beams, columns, windows, stairs and clutter).
- Object ID: A dict-mapping for any of the 14 object labels. They're the labels to be used in the PointNet model implementation.
- Health Status: Either "good" or "bad" (to avoid processing corrupted point cloud files)

The information from this custom ground truth file is mainly used to get the proper data and labels to be used by the following 2 custom datasets:

#### S3DISDataset4Classification

This is the dataset used in conjunction with the **classification** network of the PointNet architecture. 

- **Input data**: The object files (*chair_1.txt*, *table_2.txt*,...*object_Y.txt*) located inside the `Annotations` folder are going to be used as the input data for the classification network. 
- **Labels**: The label for the object data is directly extracted from the custom ground truth file *Object ID* col.

#### S3DISDataset4Segmentation

This is the dataset used for **semantic segmentation**. Since semantic segmentation needs every point in the cloud to be labeled, a new file is generated for every space/room with the suitable object labels. To do so, all files located in the `Annotations` folder are concatenated (along with the proper label) to create a single file per space/room with all the object points that belong to this space already labeled. This file is called `space_x_annotated.txt` (e.g., *office_1_annotated.txt*) and contains 7 cols (instead of 6): XYZRGB+*Object ID*. 

Since every `space_x_annotated.txt` might have millions of points, point clouds for rooms are "sliced" into smaller blocks (called *sliding windows*) to improve model training performance. 

So the S3DISDataset4Segmentation will use the contents of the sliding windows folder to get both the **input data** and **labels**

### Sliding windows

To train room segmentation, we divide each room into sections of specific dimensions, and output only the points inside said section separately from the others. 

Said sections or windows can overlap with and overlap factor going from 0%(no overlapping) to 99%(almost complete overlapping, choosing 100% overlap would lead to an infinite loop always outputting the same window).
The window width (X) and depth(Y) are specified as hyperparameters. They can be defined separately, but it makes sense that they would be the same value since objects in a room are commonly rotated on the X-Y plane.
The window height can be specified as a hyperparameter and the model is ready in case windowing in Z is necessary, but as the selected classes for segmentation are movable objects, and these are usually laid on the floor, the most logical solution is to consider all points inside a window defined by only their X-Y coordinates, and to just take all the points height-wise. The height parameter is thus ignored in the current script. This configuration would lead to the windows having a pillar-shape, from the floor to the ceiling of each room.

Having defined the parameters, we take each of the defined windows and select only the points of the room point cloud whose coordinates fall inside said windows. Because the window point clouds will be the inputs to our segmentation model, they must be independent from one another and from the room coordinates. We must then create a new reference system for every window, where the coordinates of each point refer to the origin point of each window (winX=0 winY=0) instead of to the origin of the original room point cloud.

Additionally, so the training is easier, we will normalize every window so that the point coordinates range only from 0-1.

Each window will have information on the relative coordinates of each point in it, their color, and the absolute coordinates those points had before the transformation. We also store the window identifier and the associated label. This allows us to have the spatial information if we wish to visualize the whole room results afterwards. Keep in mind that in cases where the overlap is higher than 0, some points will be in several windows at the same time, and will potentially have different predictions in each. This will have to take into account in case the implementation of a room prediction visualizer using overlap were desired.

Since the rooms are of an irregular shape, during the window sweep we might run into both empty and partially filled windows. In these cases the script will discard any window which space is not filled up to at least 90% of it's boundaries, so as to train allways with filled windows. If overlap is low, this means that some points near the walls might not be taken into account, so objects that are typically against walls might not be fully represented.

### The final folder structure 

Taking into account the previous information, the final folder structure for the model implemenation is the following one:
```
├── s3dis_summary.csv (the ground truth file)
├── Area_N
│   ├── space_X
│   │   ├── space_x.txt (the non-annotated file with the point cloud for this space. It only contains 6 cols per row: XYZRGB)
│   │   ├── space_x_annotated.txt (the annotated file with the point cloud for this space. It contains 7 cols per row: XYZRGB+*Object ID*)
│   │   ├── Annotations
│   │   │   ├── object_1.txt (the file with the point cloud for object_1 that can be found in Space_X. It contains 6 cols per row: XYZRGB)
|   |   |   ├── ...
│   │   │   ├── object_Y.txt (the file with the point cloud for object_Y that can be found in Space_X. It contains 6 cols per row: XYZRGB)
├── sliding_windows (sliced portions of all the the space_x_annotated.txt files)
│   └── w1_d1_h3_o0 
├── checkpoints 
├── runs (for TensorBoard logging)
└── tnet_outputs (output images of T-Net features)
```  


## The model

4 Python classes have been coded in the `model.py` file to implement the full PointNet architecture. Every Python class corresponds to the 
red squared section of the pictures below from the original paper.

### TransformationNet

<img width="980" alt="image" src="https://user-images.githubusercontent.com/76537012/174836557-f1a113cd-8953-4e54-bfff-da6a03be40e2.png">


#### Mathematical introduction

The T-Net is a network that estimates a affine transformation matrix. Given two affine spaces $A_1$ and $A_2$, an affine transformation, also called affinity, is a morphism between $A_1$ and $A_2$ such that the induced map $f_P: E_1 \rightarrow E_2$ with the point $P \in A_1$  is a linear map.

An affinity doesn't necessarily preserve neither the distances nor the angles but it preserves, by definition, the collinearity and the parallelism. In other words, all points belonging to a line will be alinead in, what we call canonical space, after this transformation. All the parallel lines will be preserved too. In fact, is very likely that this transformation changes the distances and the angles of our point cloud as we will see in the examples.

The output of this network will be a matrix that will be multiplied with the point cloud.


#### Topology of the network

We will present the structure of the first T-Net that appears in the network. In this case, because the number of coordinates of our point cloud is only 3, the output of the network will be a $3 \times 3$ matrix:

![tnet](https://user-images.githubusercontent.com/97680577/178104139-0f1cba1f-3e0a-4f07-a082-d0967653034f.png)


#### Visualization of the Outputs

#### Goal
The main goal of the affine transformation matrix is to make the main network learn faster. It is achieved by aligning the point clouds in a canonical way that makes it easier for the network to work with.


### BasePointNet

<img width="962" alt="image" src="https://user-images.githubusercontent.com/76537012/174834124-c25dbcfe-2c46-4616-88dd-af417c0975ee.png">


### ClassificationPointNet

<img width="980" alt="image" src="https://user-images.githubusercontent.com/76537012/174834793-8c03b4b5-933e-447b-afbb-146bb861bd13.png">


### SegmentationPointNet

<img width="980" alt="image" src="https://user-images.githubusercontent.com/76537012/174835726-32d68eaa-6196-4f1c-a545-8cfa61a28298.png">


### The data flow and model size

Using the torchinfo library, detailed information about the PointNet architecture implementation can be gathered.   

If the model input is set to (batch_size, max_points_per_space_or_object, dimensions_per_object), where: 

- batch_size = 32
- max_points_per_space_or_object = 4096
- dimensions_per_object = 3 (only xyz coordinates are taken into account. No color information is provided to the model)

the following information is shown:

#### For classification
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ClassificationPointNet                   [32, 14]                  --
├─BasePointNet: 1-1                      [32, 1024]                --
│    └─TransformationNet: 2-1            [32, 3, 3]                --
│    │    └─Conv1d: 3-1                  [32, 64, 4096]            256
│    │    └─BatchNorm1d: 3-2             [32, 64, 4096]            128
│    │    └─Conv1d: 3-3                  [32, 128, 4096]           8,320
│    │    └─BatchNorm1d: 3-4             [32, 128, 4096]           256
│    │    └─Conv1d: 3-5                  [32, 1024, 4096]          132,096
│    │    └─BatchNorm1d: 3-6             [32, 1024, 4096]          2,048
│    │    └─Linear: 3-7                  [32, 512]                 524,800
│    │    └─BatchNorm1d: 3-8             [32, 512]                 1,024
│    │    └─Linear: 3-9                  [32, 256]                 131,328
│    │    └─BatchNorm1d: 3-10            [32, 256]                 512
│    │    └─Linear: 3-11                 [32, 9]                   2,313
│    └─Conv1d: 2-2                       [32, 64, 4096]            256
│    └─BatchNorm1d: 2-3                  [32, 64, 4096]            128
│    └─TransformationNet: 2-4            [32, 64, 64]              --
│    │    └─Conv1d: 3-12                 [32, 64, 4096]            4,160
│    │    └─BatchNorm1d: 3-13            [32, 64, 4096]            128
│    │    └─Conv1d: 3-14                 [32, 128, 4096]           8,320
│    │    └─BatchNorm1d: 3-15            [32, 128, 4096]           256
│    │    └─Conv1d: 3-16                 [32, 1024, 4096]          132,096
│    │    └─BatchNorm1d: 3-17            [32, 1024, 4096]          2,048
│    │    └─Linear: 3-18                 [32, 512]                 524,800
│    │    └─BatchNorm1d: 3-19            [32, 512]                 1,024
│    │    └─Linear: 3-20                 [32, 256]                 131,328
│    │    └─BatchNorm1d: 3-21            [32, 256]                 512
│    │    └─Linear: 3-22                 [32, 4096]                1,052,672
│    └─Conv1d: 2-5                       [32, 128, 4096]           8,320
│    └─BatchNorm1d: 2-6                  [32, 128, 4096]           256
│    └─Conv1d: 2-7                       [32, 1024, 4096]          132,096
│    └─BatchNorm1d: 2-8                  [32, 1024, 4096]          2,048
├─Linear: 1-2                            [32, 512]                 524,800
├─BatchNorm1d: 1-3                       [32, 512]                 1,024
├─Linear: 1-4                            [32, 256]                 131,328
├─BatchNorm1d: 1-5                       [32, 256]                 512
├─Dropout: 1-6                           [32, 256]                 --
├─Linear: 1-7                            [32, 14]                  3,598
==========================================================================================
Total params: 3,464,791
Trainable params: 3,464,791
Non-trainable params: 0
Total mult-adds (G): 55.92
==========================================================================================
Input size (MB): 1.57
Forward/backward pass size (MB): 7652.64
Params size (MB): 13.86
Estimated Total Size (MB): 7668.08
==========================================================================================
```

#### For segmentation

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
SegmentationPointNet                     [32, 14, 4096]            --
├─BasePointNet: 1-1                      [32, 1024]                --
│    └─TransformationNet: 2-1            [32, 3, 3]                --
│    │    └─Conv1d: 3-1                  [32, 64, 4096]            256
│    │    └─BatchNorm1d: 3-2             [32, 64, 4096]            128
│    │    └─Conv1d: 3-3                  [32, 128, 4096]           8,320
│    │    └─BatchNorm1d: 3-4             [32, 128, 4096]           256
│    │    └─Conv1d: 3-5                  [32, 1024, 4096]          132,096
│    │    └─BatchNorm1d: 3-6             [32, 1024, 4096]          2,048
│    │    └─Linear: 3-7                  [32, 512]                 524,800
│    │    └─BatchNorm1d: 3-8             [32, 512]                 1,024
│    │    └─Linear: 3-9                  [32, 256]                 131,328
│    │    └─BatchNorm1d: 3-10            [32, 256]                 512
│    │    └─Linear: 3-11                 [32, 9]                   2,313
│    └─Conv1d: 2-2                       [32, 64, 4096]            256
│    └─BatchNorm1d: 2-3                  [32, 64, 4096]            128
│    └─TransformationNet: 2-4            [32, 64, 64]              --
│    │    └─Conv1d: 3-12                 [32, 64, 4096]            4,160
│    │    └─BatchNorm1d: 3-13            [32, 64, 4096]            128
│    │    └─Conv1d: 3-14                 [32, 128, 4096]           8,320
│    │    └─BatchNorm1d: 3-15            [32, 128, 4096]           256
│    │    └─Conv1d: 3-16                 [32, 1024, 4096]          132,096
│    │    └─BatchNorm1d: 3-17            [32, 1024, 4096]          2,048
│    │    └─Linear: 3-18                 [32, 512]                 524,800
│    │    └─BatchNorm1d: 3-19            [32, 512]                 1,024
│    │    └─Linear: 3-20                 [32, 256]                 131,328
│    │    └─BatchNorm1d: 3-21            [32, 256]                 512
│    │    └─Linear: 3-22                 [32, 4096]                1,052,672
│    └─Conv1d: 2-5                       [32, 128, 4096]           8,320
│    └─BatchNorm1d: 2-6                  [32, 128, 4096]           256
│    └─Conv1d: 2-7                       [32, 1024, 4096]          132,096
│    └─BatchNorm1d: 2-8                  [32, 1024, 4096]          2,048
├─Conv1d: 1-2                            [32, 512, 4096]           557,568
├─BatchNorm1d: 1-3                       [32, 512, 4096]           1,024
├─Conv1d: 1-4                            [32, 256, 4096]           131,328
├─BatchNorm1d: 1-5                       [32, 256, 4096]           512
├─Conv1d: 1-6                            [32, 128, 4096]           32,896
├─BatchNorm1d: 1-7                       [32, 128, 4096]           256
├─Conv1d: 1-8                            [32, 14, 4096]            1,806
==========================================================================================
Total params: 3,528,919
Trainable params: 3,528,919
Non-trainable params: 0
Total mult-adds (G): 150.75
==========================================================================================
Input size (MB): 1.57
Forward/backward pass size (MB): 9545.98
Params size (MB): 14.12
Estimated Total Size (MB): 9561.66
==========================================================================================
```

### Metrics

#### For Classification

##### $F_1$ Score
First of all we need to define what precision and recall are:

$$precision = \frac{TP}{TP+FP}$$

$$recall=\frac{TP}{TP+FN}$$

We can define the $F_1$ Score as the harmonic mean of the precision and the recall:

$$F_1=2\frac{precision\times recall}{precision+recall}=\frac{TP}{TP+\frac{1}{2}(FP+FN)}$$


##### Area Under the Curve (AUC)
The ROC curve is the curve that represents the relation between True Positive Rate and False Positive Rate:
$$TPR=\frac{TP}{TP+FN}=recall$$
$$FPR=\frac{FP}{FP+TN}=1-precision$$

The AUC metric stands for the integral of this curve between 0 and 1.

![AUC](https://user-images.githubusercontent.com/97680577/178116164-b8a46799-99c5-4d8d-8a1b-44a7c9ed912f.png)

The AUC metric tells you how capable is a moder to distinguish between classes. A model that have a measure near to 1 means that it has a good separability. In the following example we will see a model whose ROC curve has an area of 1.

![1_HmVIhSKznoW8tFsCLeQjRw](https://user-images.githubusercontent.com/97680577/178118715-d8130218-05aa-4aa2-94e4-08f4463c2953.png)

In other words, a threashold that distinguish between two classes can easily be found, creating no FN nor FP.

![1_Uu-t4pOotRQFoyrfqEvIEg](https://user-images.githubusercontent.com/97680577/178118753-a3064e6d-2215-4113-b2cf-452957551b3b.png)

Nevertheless a model whose separability is not that good, might have a ROC curve like the following one:

![1_-tPXUvvNIZDbqXP0qqYNuQ](https://user-images.githubusercontent.com/97680577/178118821-b6bad0de-4d02-41be-a8d6-d408ccc449ce.png)

So if we visualise the threshold it might be like this:

![1_yF8hvKR9eNfqqej2JnVKzg](https://user-images.githubusercontent.com/97680577/178118843-f6c30a95-9c60-4887-aef0-7ee05915d5a2.png)

#### For Segmentation

##### IoU Score (Intersection over Union):
When we are dealing with a Segmentation problem, not only we need to have in consideration the pixels that we labeled wrongly (false positives) but we need to consider the pixels belonging to the class that we didn't label (false negatives). 


$$IoU = \frac{|A\cap B|}{|A \cup B|} = \frac{TP}{TP+FN+FP} $$

![IoU](https://user-images.githubusercontent.com/97680577/178110909-c405e44c-74a9-404f-a355-dad7cedea66e.png)

Considering the green rectangle the correct ones and the red rectangle the prediction.



## Main Conclusions

Segmentation:

- When very few points are used (i.e., 100 points per space), only walls are learned to be detected
- When very few points are used it's not convenient to use RGB data (revision)

## Related Work

## Contributors
This is section1 

## Acknowledgments
This is section1 


