# 3D-semantic-parsing
Repo to host the UPC AIDL spring 2022 post-graduate project

## Table of Contents
- [Abstract](#abstract)
- [Main goals](#main-goals)
- [The dataset](#the-dataset)
  * [The custom ground truth file](#the-custom-ground-truth-file)
    + [S3DISDataset4Classification](#s3disdataset4classification)
    + [S3DISDataset4Segmentation](#s3disdataset4segmentation)
    + [Discarding non-movable classes for segmentation](#discarding-non-movable-classes-for-segmentation)
    + [Sliding windows for segmentation](#sliding-windows)
    + [Discarding inadequate windows](#discarding-inadequate-windows)
    + [Sampling rate](#sampling-rate)
  * [The final folder structure](#the-final-folder-structure)
- [The model](#the-model)
  * [TransformationNet](#transformationnet)
    + [Mathematical introduction](#mathematical-introduction)
    + [Topology of the network](#topology-of-the-network)
    + [Visualization of the Outputs](#visualization-of-the-outputs)
    + [Goal](#goal)
  * [BasePointNet](#basepointnet)
  * [ClassificationPointNet](#classificationpointnet)
  * [SegmentationPointNet](#segmentationpointnet)
  * [The data flow and model size](#the-data-flow-and-model-size)
    + [For classification](#for-classification)
    + [For segmentation](#for-segmentation)
- [The metrics](#metrics)
  * [F1 for classification](#f1-for-classification)
  * [IoU for segmentation](#iou-for-segmentation)
- [Main conclusions](#main-conclusions)
- [How to run the code](#how-to-run-the-code)
  * [Download the S3DIS dataset](#download-the-s3dis-dataset)
  * [Create a conda virtual environment](#create-a-conda-virtual-environment)
  * [Running the code](#running-the-code)
- [Related work](#related-work)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)
- [Annex](#annex)

## Abstract
A point cloud is a type of 3D geometric data structure, based on an unordered set of points.
Each point is a vector of its (x, y, z) coordinates plus extra feature channels such as color, normal, etc.

This project is based in the [PointNet](https://arxiv.org/abs/1612.00593?context=cs) architecture, which provides a unified deep learning architecture for the following recognition tasks, based on point cloud inputs:
1) Shape classification 
2) Shape part segmentation
3) Scene semantic parsing

This project'll be focus on implementing only **object classification** and **scene semantic parsing** (no shape part segmentation is carried out). As a dataset, the **S3DIS dataset** is going to be used, where every point includes both its spatial coordinates and its color info (xyzrgb).


## Main goals
The general stategy is to implement a PointNet architecture in Pytorch that uses the S3DIS dataset in order to perform object classification and indoor scene semantic segmentation. 

We will mainly focus on the 5 movable objects presented in the dataset (board, bookcase, chair, table and sofa) to carry out the following tasks:

- **Classification** of movable elements given its own point cloud. 
- **Semantic segmentation** of each object given a room point cloud.
- Study the **impact on accuracy metrics** of:
   - The inclusion of ***clutter*** objects 
   - The inclusion of **color**
   - The size of the ***sliding windows***
   - **Discarding strategies** during *sliding windows* creation 
   - *Sliding windows* **overlapping**
   - The **sampling rate** (number of points taken into account per object/room) during dataset creation
   - The amount of epochs
- Study **correlation** between metrics and actual semantic segmenatation outcomes.
    

## The dataset
The **3D Semantic Parsing of Large-Scale Indoor Spaces (S3DIS)** dataset is going to be used in order to work with the PointNet architecture. 

The **basic dataset statistics** are:

- It contains point clouds from **3 different buildings**, distributed in **6 areas**:
  - Building 1: Area_1, Area_3, Area_6 
  - Building 2: Area_2, Area_4 
  - Building 3: Area_5 
- There are **11 different room types** (WCs, conference rooms, copy rooms, hallways, offices, pantries, auditoriums, storage rooms, lounges, lobbies and open spaces) for a grand total of  **272 rooms** (or spaces) distributed among all areas.  
- Every room can have up to **14 different object types**. These objects can be classified as:
  - **movable** (boards, bookcases, chairs, tables and sofas)  
  - **structural** (ceilings, doors, floors, walls, beams, columns, windows and stairs). 
  - **clutter** (if an object doesn't belong to any of the previous categories)

More **advanced statistics** can be found after a slightly deeper analysis:

|<img width="1040" alt="image" src="https://user-images.githubusercontent.com/76537012/175812289-8b537a96-cb85-425e-a47c-5120a31baf9b.png">|
|---|
| <sub> Object classes dict: {0:ceiling, 1:clutter, 2:door, 3:floor, 4:wall, 5:beam, 6:board, 7:bookcase, 8:chair, 9:table, 10:column, 11:sofa, 12:window, 12:window, 13:stairs } </sub>|
| <sub> Room types dict: {0:WC, 1:conferenceRoom, 2:copyRoom, 3:hallway, 4:office, 5:pantry, 6:auditorium, 7:storage, 8:lounge, 9:lobby, 10:openSpace}</sub>|

Some illustrative rough figures, from aproximately **273M** total points:

| Object           | Ceiling | Door | Floor | Wall | Beam | Board | Bookcase | Chair | Table | Column | Sofa | Window | Stairs| Clutter
|:----------------:|:-------:|:----:|:-----:|:----:|:----:|:-----:|:--------:|:-----:|:-----:|:-----:|:-----:|:------:|:-----:|:-----:|
| Total points(M)  | 53      | 13   | 43    | 76   |  5   | 3     | 17       |    9  | 9     | 5     | 1     | 7      |  0.6  | 28


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

- `object_Y.txt` is a file containing the point cloud of this particular object that belongs to Space_X (e.g., objects `chair_1.txt`, `chair_2.txt` or `table_1.txt` from a space/room called office_1). This point cloud file has 6 columns (non-normalized XYZRGB values).
- `space_x.txt` is a non-annotated point cloud file containing the sum of all the point cloud object files (object_1, object_2, ...) located within the `Annotations` folder (e.g., the space file called `Area_1\office_1\office_1.txt` contains the sum of the object files `chair_1.txt`, `chair_2.txt` and the rest of all the object files located inside the `Annotations` directory). As a consequence, the space/room point cloud file has only 6 columns too (non-normalized XYZRGB values).

Comprehensive information about the original S3DIS dataset can be found at: http://buildingparser.stanford.edu/dataset.html 

From this original S3DIS dataset:

- A custom ground truth file (called `s3dis_summary.csv`) has been created to speed up the process to get to the point cloud files, avoiding recurrent operating system folder traversals.  
- Two custom datasets have been created to feed the dataloaders, depending on the desired goal (S3DISDataset4Classification and S3DISDataset4Segmentation). 
- Since dataloaders expect the same amount of input points but rooms/objects might differ considerably, a user-defined threshold can be set to limit the number of points to sample per room/object when datasets are created. This threshold is specified in the *hparams* dictionary from the `settings.py` file.
- The available areas have been splitted and assigned to the following tasks:
  - Training: Areas 1, 2, 3 and 4
  - Validation: Area 5
  - Test: Area 6

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

- **Input data**: The object files (`chair_1.txt`, `table_2.txt`,...`object_Y.txt`) located inside the `Annotations` folder are going to be used as the input data for the classification network. 
- **Labels**: The label for the object data is directly extracted from the custom ground truth file *Object ID* col.

```
--------------------------------------------------------------------------------
S3DIS DATASET INFORMATION (S3DISDataset4ClassificationTrain)
--------------------------------------------------------------------------------
Summary file: s3dis_summary_clutter_free_movable.csv
Data source folder: /Users/jgalera/datasets/S3DIS/aligned 
Chosen areas: ['Area_1', 'Area_2', 'Area_3', 'Area_4'] 
Dataset elements:
           0                 1               2  3               4      5         6  7     8
0     Area_1  conferenceRoom_1  conferenceRoom  1     board_1.txt  19727     board  0  Good
1     Area_1  conferenceRoom_1  conferenceRoom  1     board_2.txt  33747     board  0  Good
2     Area_1  conferenceRoom_1  conferenceRoom  1     board_3.txt  33359     board  0  Good
3     Area_1  conferenceRoom_1  conferenceRoom  1  bookcase_1.txt  20437  bookcase  1  Good
4     Area_1  conferenceRoom_1  conferenceRoom  1     chair_1.txt   6729     chair  2  Good
...      ...               ...             ... ..             ...    ...       ... ..   ...
1522  Area_4         storage_4         storage  7  bookcase_1.txt  20372  bookcase  1  Good
1523  Area_4         storage_4         storage  7  bookcase_2.txt  15859  bookcase  1  Good
1524  Area_4         storage_4         storage  7  bookcase_3.txt   6882  bookcase  1  Good
1525  Area_4         storage_4         storage  7  bookcase_4.txt  16144  bookcase  1  Good
1526  Area_4         storage_4         storage  7  bookcase_5.txt  43708  bookcase  1  Good

[1527 rows x 9 columns]

--------------------------------------------------------------------------------
S3DIS DATASET INFORMATION (S3DISDataset4ClassificationVal)
--------------------------------------------------------------------------------
Summary file: s3dis_summary_clutter_free_movable.csv
Data source folder: /Users/jgalera/datasets/S3DIS/aligned 
Chosen areas: ['Area_5'] 
Dataset elements:
          0                 1               2  3               4      5         6  7     8
0    Area_5  conferenceRoom_1  conferenceRoom  1     board_1.txt  23363     board  0  Good
1    Area_5  conferenceRoom_1  conferenceRoom  1     chair_1.txt   4112     chair  2  Good
2    Area_5  conferenceRoom_1  conferenceRoom  1     chair_2.txt   4002     chair  2  Good
3    Area_5  conferenceRoom_1  conferenceRoom  1     chair_3.txt   2299     chair  2  Good
4    Area_5  conferenceRoom_1  conferenceRoom  1     chair_4.txt   4548     chair  2  Good
..      ...               ...             ... ..             ...    ...       ... ..   ...
677  Area_5         storage_2         storage  7  bookcase_7.txt  22424  bookcase  1  Good
678  Area_5         storage_4         storage  7     chair_1.txt   6068     chair  2  Good
679  Area_5         storage_4         storage  7     chair_2.txt   4484     chair  2  Good
680  Area_5         storage_4         storage  7     table_1.txt   8385     table  3  Good
681  Area_5         storage_4         storage  7     table_2.txt   6469     table  3  Good

[682 rows x 9 columns]

--------------------------------------------------------------------------------
S3DIS DATASET INFORMATION (S3DISDataset4ClassificationTest)
--------------------------------------------------------------------------------
Summary file: s3dis_summary_clutter_free_movable.csv
Data source folder: /Users/jgalera/datasets/S3DIS/aligned 
Chosen areas: ['Area_6'] 
Dataset elements:
          0                 1               2   3             4      5      6  7     8
0    Area_6  conferenceRoom_1  conferenceRoom   1   board_1.txt  25892  board  0  Good
1    Area_6  conferenceRoom_1  conferenceRoom   1   board_2.txt  19996  board  0  Good
2    Area_6  conferenceRoom_1  conferenceRoom   1   chair_1.txt   6548  chair  2  Good
3    Area_6  conferenceRoom_1  conferenceRoom   1  chair_10.txt   4603  chair  2  Good
4    Area_6  conferenceRoom_1  conferenceRoom   1   chair_2.txt   5555  chair  2  Good
..      ...               ...             ...  ..           ...    ...    ... ..   ...
379  Area_6       openspace_1       openspace  10   table_1.txt  34368  table  3  Good
380  Area_6       openspace_1       openspace  10   table_2.txt  35873  table  3  Good
381  Area_6       openspace_1       openspace  10   table_3.txt  33109  table  3  Good
382  Area_6          pantry_1          pantry   5   table_1.txt   5200  table  3  Good
383  Area_6          pantry_1          pantry   5   table_2.txt  22381  table  3  Good

[384 rows x 9 columns]
```

#### S3DISDataset4Segmentation

This is the dataset used for **semantic segmentation**. Since semantic segmentation needs every point in the cloud to be labeled, a new file is generated for every space/room with the suitable object labels. To do so, all files located in the `Annotations` folder are concatenated (along with the proper label) to create a single file per space/room with all the object points that belong to this space already labeled. This file is called `space_x_annotated.txt` (e.g., `office_1_annotated.txt`) and contains 7 cols (instead of 6): XYZRGB+*Object ID*.

Since every `space_x_annotated.txt` might have millions of points, point clouds for rooms are "sliced" into smaller blocks (called *sliding windows*) to improve model training performance. The slicing is a pre-process carried out only once per defined sliding window settings. The slices will be saved as Pytorch files (\*.pt) inside the `sliding_windows` folder.

So the S3DISDataset4Segmentation will use the contents of the sliding windows folder to get both the **input data** and **labels**

```
----------------------------------------------------------------------------------------------------
S3DIS DATASET INFORMATION (S3DISDataset4SegmentationTrain)
----------------------------------------------------------------------------------------------------
Summary file: s3dis_summary_clutter_free_movable.csv
Data source folder: /Users/jgalera/datasets/S3DIS/aligned 
Chosen areas: ['Area_1', 'Area_2', 'Area_3', 'Area_4'] 
Total points (from sliding windows): 5925067 
Total points (after sampling sliding windows at 768 points/room): 7715 
Total dataset elements: 472 (from a grand total of 883)
From Area_1 : 118 (['Area_1_conferenceRoom_1_win0.pt', 'Area_1_conferenceRoom_1_win1.pt', 'Area_1_conferenceRoom_1_win2.pt']...['Area_1_office_9_win2.pt', 'Area_1_office_9_win3.pt', 'Area_1_office_9_win4.pt'])
From Area_2 : 193 (['Area_2_auditorium_1_win0.pt', 'Area_2_auditorium_1_win1.pt', 'Area_2_auditorium_1_win10.pt']...['Area_2_office_5_win3.pt', 'Area_2_office_5_win4.pt', 'Area_2_office_5_win5.pt'])
From Area_3 : 66 (['Area_3_conferenceRoom_1_win0.pt', 'Area_3_conferenceRoom_1_win1.pt', 'Area_3_conferenceRoom_1_win2.pt']...['Area_3_office_9_win0.pt', 'Area_3_storage_1_win0.pt', 'Area_3_storage_1_win1.pt'])
From Area_4 : 95 (['Area_4_conferenceRoom_1_win0.pt', 'Area_4_conferenceRoom_1_win1.pt', 'Area_4_conferenceRoom_1_win2.pt']...['Area_4_office_9_win4.pt', 'Area_4_office_9_win5.pt', 'Area_4_storage_1_win0.pt'])

----------------------------------------------------------------------------------------------------
S3DIS DATASET INFORMATION (S3DISDataset4SegmentationVal)
----------------------------------------------------------------------------------------------------
Summary file: s3dis_summary_clutter_free_movable.csv
Data source folder: /Users/jgalera/datasets/S3DIS/aligned 
Chosen areas: ['Area_5'] 
Total points (from sliding windows): 4496525 
Total points (after sampling sliding windows at 768 points/room): 5855 
Total dataset elements: 261 (from a grand total of 883)
From Area_5 : 261 (['Area_5_conferenceRoom_1_win0.pt', 'Area_5_conferenceRoom_1_win1.pt', 'Area_5_conferenceRoom_1_win2.pt']...['Area_5_office_9_win2.pt', 'Area_5_office_9_win3.pt', 'Area_5_storage_2_win0.pt'])

----------------------------------------------------------------------------------------------------
S3DIS DATASET INFORMATION (S3DISDataset4SegmentationTest)
----------------------------------------------------------------------------------------------------
Summary file: s3dis_summary_clutter_free_movable.csv
Data source folder: /Users/jgalera/datasets/S3DIS/aligned 
Chosen areas: ['Area_6'] 
Total points (from sliding windows): 1803057 
Total points (after sampling sliding windows at 768 points/room): 2348 
Total dataset elements: 150 (from a grand total of 883)
From Area_6 : 150 (['Area_6_conferenceRoom_1_win0.pt', 'Area_6_conferenceRoom_1_win1.pt', 'Area_6_conferenceRoom_1_win2.pt']...['Area_6_office_9_win6.pt', 'Area_6_openspace_1_win0.pt', 'Area_6_openspace_1_win1.pt'])
```

#### Discarding non-movable classes for segmentation

The original S3DIS dataset comes with some non-movable classes (structural and clutter defined above).

In order to make the model learn faster and better, only the movable objects have been taking into account for semantic segmentation. The hypothesis is that removing this information will allow the model to focus on the target classes and prevent it from focusing on classes that are not of interest. 

![image](https://user-images.githubusercontent.com/104381341/178322532-8e1e77e2-0ad8-4673-a685-bda9f3718932.png)

#### Sliding windows for segmentation

To train, validate and test room segmentation, we divide each room into sections of specific dimensions, and output only the points inside said section separately from the others. 

![image](https://user-images.githubusercontent.com/104381341/178322824-590e4aa2-8962-4298-babe-0069f76de9b1.png)

The window width (X) and depth(Y) are specified as hyperparameters. They can be defined separately, but it makes sense that they would be the same value since objects in a room are commonly rotated on the X-Y plane.
The window height can be specified as a hyperparameter and the model is ready in case windowing in Z is necessary, but as the selected classes for segmentation are movable objects, and these are usually laid on the floor, the most logical solution is to consider all points inside a window defined by only their X-Y coordinates, and to just take all the points height-wise. The height parameter is thus ignored in the current script. This configuration would lead to the windows having a pillar-shape, from the floor to the ceiling of each room.

![image](https://user-images.githubusercontent.com/104381341/178575853-34dc54b6-9925-48a4-bc98-627e284715e7.png)

Said sections or windows can overlap with and overlap factor going from 0%(no overlapping) to 99%(almost complete overlapping, choosing 100% overlap would lead to an infinite loop always outputting the same window). The efect of this variable will be considered in the study

![image](https://user-images.githubusercontent.com/104381341/178322944-865dc454-cac9-44d2-b501-0d1550f533b1.png)

Having defined the parameters, we take each of the defined windows and select only the points of the room point cloud whose coordinates fall inside said windows. 

Because the window point clouds will be the inputs to our segmentation model, they must be independent from one another and from the room coordinates. We must then create a new reference system for every window, where the coordinates of each point refer to the origin point of each window (winX=0 winY=0) instead of to the origin of the original room point cloud. 

Additionally, so the training is easier, we will normalize every window so that the point coordinates range only from 0-1.

Each window will have information on the relative coordinates of each point in it, their color, and the absolute coordinates those points had before the transformation. We also store the window identifier and the associated label. This allows us to have the spatial information if we would wish to visualize the whole room results afterwards. Keep in mind that in cases where the overlap is higher than 0, some points will be in several windows at the same time, and will potentially have different predictions in each. This will have to be taken into account in case the implementation of a room prediction visualizer using overlap were desired.

The structure of each individual window file will be: 

| Rel. Coord.   |   RGB         | Abs. Coord   | Window ID  | Object Label |
|:-------------:|:-------------:|:------------:|:----------:|:------------:|
|   [3 x int]   |    [3 x int]  |   [3 x int]  | integer    |    [0-5]     | 

#### Discarding inadequate windows

Since the rooms are of an irregular shape, during the window sweep we might run into both empty and partially filled windows.

In the first case, the script will discard any window that is completely empty.

For the second case, if one of the resulting windows has at least one point, we have put in place a strategy that allows us to select a desirable percentage of "window filling". If we wanted the window to be at least 80% filled, the script will create the window, find the coordinates of the points that are further to the left, further to the right further to the front and further to the back of said window, and find the distances betweem them (left-right, front-back). If one of those distances is smaller than 80% of the window size, the window wil be discarded, considered not filled enough. The default is 90% filled. This variable will be studied

#### Sampling rate

For the PointNet to work, the dimensions of all the inputs must be the same. However both object point clouds for classification and sliding window point clouds for segmentation have different number of points. Hence, prior to entering the data in the model, these point clouds must be modified to fit this variable. At the same time, this is one of the effects that will be studied as a hyperparameter.

### The final folder structure 

Taking into account all the above information, the final folder structure for the model implemenation is the following one:
```
├── s3dis_summary.csv (the ground truth file)
├── s3dis_summary_clutter_free_movable.csv (the ground truth file containing info only for movable objects)
├── Area_N
│   ├── space_X
│   │   ├── space_x.txt (the original non-annotated file with the point cloud for this space. It only contains 6 cols per row: XYZRGB)
│   │   ├── space_x_annotated.txt (the annotated file with the point cloud for this space (including clutter) It contains 7 cols per row: XYZRGB+*Object ID*)
│   │   ├── space_x_annotated_clutter_free.txt (the annotated file for this space (excluding clutter). It contains 7 cols per row: XYZRGB+*Object ID*)
│   │   ├── Annotations
│   │   │   ├── object_1.txt (the file with the point cloud for object_1 that can be found in Space_X. It contains 6 cols per row: XYZRGB)
|   |   |   ├── ...
│   │   │   ├── object_Y.txt (the file with the point cloud for object_Y that can be found in Space_X. It contains 6 cols per row: XYZRGB)
├── sliding_windows (sliced portions of all the the space_x_annotated.txt files)
│   └── w1_d1_h3_o0 
├── checkpoints 
├── cameras (to store required JSON files to visualize segmentation with Open3D)
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

An affinity doesn't necessarily preserve neither the distances nor the angles, but it preserves, by definition, the collinearity and the parallelism. In other words, all points belonging to a line will be aligned in, what we call canonical space, after this transformation. All the parallel lines will be preserved too. In fact, it is very likely that this transformation changes the distances and the angles of our point cloud as we will see in the examples.

The output of this network will be a matrix that will be multiplied with the point cloud.


#### Topology of the network

We will present the structure of the first T-Net that appears in the network. In this case, because the number of coordinates of our point cloud is only 3, the output of the network will be a $3 \times 3$ matrix:

![tnet](https://user-images.githubusercontent.com/97680577/178104139-0f1cba1f-3e0a-4f07-a082-d0967653034f.png)


#### Visualization of the outputs

It's important to mention that the T-Net, like any Machine Learning model, performs as well as the network is trained. This means that if the model is not properly trained, it cannot be guaranteed that the model is invariant to rigid transformations. 

#### Goal

When we are dealing with point clouds, it is normal that our data undergoes some geometric transformations. The purpose of the T-Net is to align all the point cloud in a canonical way, so it is invariant to these transformations. After doing that, feature extraction can be done. It is important to remark that in this part of the network we are using the T-Net only to make the points invariant to other transformations, so if our dataset gives more information about the points (for example the color) it won't be added to the T-Net (so the input will have 3 channels). 

When the affine transformation matrix is used again, it is not used directly in the point cloud, but in the features that had been extracted before. So in this case we are in a high dimensional space, and it is possible that we have some optimization problems. In order to avoid those, it is added a regularization term in the softmax training loss so the transformation matrix is close to the orthogonal matrix.

$$L_{reg} = ||I-AA^T||_{F}^2 $$

Where A is the transformation matrix predicted by the T-Net.

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

#### F1 for classification

First of all we need to define what precision and recall are:

$$precision = \frac{TP}{TP+FP}$$

$$recall=\frac{TP}{TP+FN}$$

We can define the $F_1$ Score as the harmonic mean of the precision and the recall:

$$F_1=2\frac{precision\times recall}{precision+recall}=\frac{TP}{TP+\frac{1}{2}(FP+FN)}$$

#### IoU for segmentation

When we are dealing with a segmentation problem, we don't only need to take into account the points that we labeled wrongly (false positives) but also the points belonging to the class that we didn't properly labeled (false negatives). 

$$IoU = \frac{|A\cap B|}{|A \cup B|} = \frac{TP}{TP+FN+FP} $$

![IoU](https://user-images.githubusercontent.com/97680577/178110909-c405e44c-74a9-404f-a355-dad7cedea66e.png)

Considering the green rectangle the correct ones and the red rectangle the prediction.

#### Micro, Macro and Weighted metrics

For both F1 and IoU metrics, micro, macro and weighted ponderations have been calculated.

Having imbalanced datasets can add some distortion to the metrics. In order to minimize it, several ponderation methods can be used:

- **Macro**: Calculate the metrics for every individual class of the sample and then average them. The problem of doing it in this way, is that if we have an imbalanced dataset with a class that contains a lot of samples, the metric result of this class will be treated as the metric result of the other classes, where we don't have as many of samples. For example, if class B has considerably more samples than class A, but class A has a much better accuracy, then the accuracy of class A will compensate the bad performance of the accuracy of class B, where we will find lots of samples incorrectly classified.

- **Micro**: Consider all the samples of all the classes at the same time. By doing so, if we had the imbalanced dataset that we described before, calculating the metric like this would expose the bad performance of the model in this imbalanced dataset.

- **Weighted Average**: Calculate the metrics similarly as the micro but considering the support (the support of the class is the number of samples of this class divided by the number of total samples of the dataset) of each class to the dataset.

### Problems with the metrics
Choosing the right metric for a specific Deep Learning task is not trivial. In our case, we chose the **IoU** metric for segmentation, but it might have some problems:
As it is a metric that is calculated from the **Confusion Matrix,** if the model doesn't map a point to a specific class it doesn't count as an incorrect prediction so
we can have high IoU metrics that don't perform well.

For example:
Here we have a room with multiple classes:

![ground_truth](https://user-images.githubusercontent.com/97680577/178546549-0e6a0917-fcb0-4963-8088-4f69472d508d.PNG)

but if we check the confussion matrix we get:

| Object   | board | bookcase | chair | table | sofa |
|----------|-------|----------|-------|-------|------|
| board    | 12685 | 0        | 0     | 0     | 0    |
| bookcase | 0     | 0        | 0     | 0     | 0    |
| chair    | 0     | 0        | 1087  | 0     | 0    |
| table    | 0     | 0        | 0     | 0     | 0    |
| sofa     | 0     | 0        | 0     | 0     | 0    |

and the metrics are:

| Score | Macro | Micro | Weighted |
|-------|-------|-------|----------|
| F1    | 1     | 1     | 1        |
| IoU   | 1     | 1     | 1        |

but as we can see the results are not as good as we might expect:

![pred](https://user-images.githubusercontent.com/97680577/178548230-2db7c2bf-fe93-4af2-aef7-a50eab6d2d4e.png)

If we check the [Scikit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html) we can see that it explains that this problem can happen.

## Main Conclusions

Classification:

- A low sampling rate (128 points/object) leads to good accuracy metrics.

Segmentation:

- About dataset preparation and discard:
   - Not implementing the discard of non-movable classes leads to the model learning only structural classes (i.e. walls, specially if very few points are used) if the original dataset is kept or, if the structural points are transformed into "clutter" points, to the model learning to identify clutter but not the rest of the classes. The strategy of discarding all non-movable points is then correct.
   - Changing the "window filling" parameter from 90% to 1% diminishes accuracy. The explanation is that if we take windows that might only have a small part of an object, the model finds it harder to identify those objects than if we already give them windows that contain the majority of an object. The same way a person would find it harder to separate a table leg from a chair leg if we only had that information, than to separate half a chair from half a table. There is probably a sweet spot in this parameter, related to window size.
   - However, the script also discards windows that might contain a full object even if the window is not completely filled. For example narrow objects like boards and bookcases or objects that might be against a wall. When we visualize the results and compare them to the ground truth we see that those objects where not even considered, the window system discarded them. 

- About RGB information:
   - RGB information is only useful when the model is in a "sweet spot". In cases where weighted IoU is over 0.45, RGB increases the value by 10%. Else it can hinder training. This prevails when the model has a high number of points, so the hypothesis that there is too much to learn (RGB on top of everything else) from too little information (number of points) does not apply.

- About window size:
   - Increasing window size from 1 to 2 leads to poor results, even when the number of points is adapted so that the "density" is equivalent. 
   - Depending on both window size and overlap, the number of points considered the optimal point varies. For window size 1 with 50% overlap, 128 already leads to almost the best results. For window size 1 with 0% overlap, the number is 512. This makes sense since with 0.5 overlap we are increasing the number of input windows by two, so it's sensible to think that we might need less points per window.
   - 
    
- About overlap:
   - As expected, overlap of 50% achieves the best results. Although it slightly increases the time of dataset preparation (done only once) and the time of training (since there are more input windows), it also allows to have best results even with a few points.
   
- About number of epochs:
   - More epochs achieve better accuracy metrics for the same sampling rate (as expected). However, the overall model performance seems to depend intrinsically of the sampling rate (and/or its relationship with the size of the sliding windows) 
   
     Epoch | 256     | 512     | 768    | 1024   | 2048
     |:----:|:-------:|:-------:|:------:|:------:|:----|
     |3  | 0.2727  | 0.2959  | 0.2746 | 0.2778 | 0.3056
     10  | 0.3562  | 0.2437  | 0.5540 | 0.3249 | 0.3464
     20  | 0.5498  | 0.5154  | 0.5626 | 0.5290 | 0.4617

     (Table X. IoU weighted scores per epoch per sampling rate over the test dataset)

- About correlation between IoU scores and actual visual segmentation outputs:
   - To be filled with some pictures
     
- General results:
  - We get the best results/cost with:
      - 128 points
      - 50% overlap
      - RGB 
      - 90% window filling discard criteria
      - Window size =1
      
   - The model seems to present a high bias for tables and chairs. This is possibly due to the window discard strategy. This needs to be worked on.
 
 
     Confusion Matrix

      Object  | board | bookcase | chair  | table  | sofa 
      |:--------:|------:|---------:|:------:|:------:|:----:|
      |  board   |  1558 |    63    |  5726  | 13070  |  6   |
      | bookcase |  575  |    75    | 24807  | 24268  |  59  |
      |  chair   |  378  |   295    | 167682 | 52964  | 988  |
      |  table   |  511  |   213    | 71347  | 339137 | 610  |
      |   sofa   |  174  |    0     | 12829  |  7062  | 3923 |



      Scores (per object)

      Scores | board | bookcase | chair | table |sofa
      |:--------:|------:|---------:|:------:|:------:|:----:|
      | Precision | 0.4875 |  0.1161  | 0.5938 | 0.7769 | 0.7023 |
      |   Recall  | 0.0763 |  0.0015  | 0.7543 | 0.8235 | 0.1635 |
      |  F1 Score | 0.1319 |  0.0030  | 0.6645 | 0.7996 | 0.2653 |

      Scores (averages)

      Score | Macro  | Micro  | Weighted 
     |:--------:|------:|---------:|:------:|
     |  IoU  | 0.2777 | 0.5426 |  0.5356  |

  - On the other side, similar IoU results are obtained with lower overlap but higher number of points and smaller windows:
	- 512 points
	- 25% overlap
	- No RGB 
	- 90% Window filling discard criteria
	- Window size =0.25

	Confusion Matrix
	| Object   | board | bookcase | chair | table | sofa |
	|----------|-------|----------|-------|-------|------|
	| board    | 0     | 0        | 0     | 87    | 0    |
	| bookcase | 0     | 0        | 48    | 4950  | 0    |
	| chair    | 0     | 0        | 10932 | 2804  | 0    |
	| table    | 0     | 0        | 3555  | 23701 | 0    |
	| sofa     | 0     | 0        | 0     | 0     | 0    |


	Scores (per object)
	Scores | board | bookcase | chair | table |sofa
	|:--------:|------:|---------:|:------:|:------:|:----:|
	| Precision | 0.0000 |  0.0000  | 0.5651 | 0.7514 | 0.0000 |
	|   Recall  | 0.0000 |  0.0000  | 0.7959 | 0.8696 | 0.0000 |
	|  F1 Score | 0.0000 |  0.0000  | 0.6609 | 0.8062 | 0.0000 |


	Scores (averages)
	| Score | Macro | Micro | Weighted |
	|-------|-------|-------|----------|
	| F1    | 0.3668 | 0.6806 | 0.6102 |
	| IoU   | 0.2922 | 0.5158 | 0.4949  |




In this case visualization is much better, and chairs and windows are correctly detected.The fact that with similar results for IoU we get different visualization results (in this case better than in the previous selection of parameters), could be explained by the window discard strategy. Since different windows are selected when we select different window sizes and overlaps, different windows are discarded so in the end, we are comparing different ground truth data and the IoU is not comparable
														 


   


## How to run the code
### Download the S3DIS dataset

- Fork/clone this repo.
- Go to the  http://buildingparser.stanford.edu/dataset.html and download the aligned version of the S3DIS dataset.
- Once downloaded, edit your forked `settings.py` file and set the 'pc_data_path' key of the *eparams* dict to the folder you downloaded tha dataset (e.g, `/datasets/S3DIS/aligned`)

### Create a conda virtual environment

Install conda (or miniconda) and follow the usual directions to create and switch to a new conda virtual environment (replace `project_name` with the name you want to give to your virtual env):

```
conda create -n project_name python=3.8.1
conda activate project_name
pip install -r requirements.txt
```

### Running the code

The code supports multiple arguments to be parsed, depending on:

- The **task** to be performed: either train, validation, test or watch.
- The **goal**: either classification or segmentation.
- The target **objects** we want to work with: either all objects or only the movable objects.
- The **load** profile: either toy, low, medium or high.

So run the code from the previously created virtual environment with the following command:
```
python main.py --task {train, validation, test, watch} --goal {classification, segmentation} --load *toy, low, medium, high} --objects {all, movable}
```
The load profiles include the following settings by default:

| Load profile | Num points per object/room (class/seg) | Epochs | Dimensions per object
|:-------------:|:-------------------------------------:|:------:|:----------------------:|
| Toy        | 10                                       | 1      | 3
| Low        | 128                                      | 3      | 3
| Medium     | 512                                      | 10     | 3
| High       | 1024                                     | 40     | 3 

(Note: The *toy* load profile is mainly intended to quickly test buggy behaviours during code development)

All these args are specified in the file `settings.py` and can be freely changed to meet your needs. 


## Related Work
1. Benjamín Gutíerrez-Becker and Christian Wachinger. _Deep Multi-Structural Shape Analysis:Application to Neuroanatomy_. 2018
2. 

## Contributors

Marc Casals i Salvador

Lluís Culí

Javier Galera

Clara Oliver

## Acknowledgments
We'd like to thank the unconditional support of our advisor Mariona Carós, whose kind directions helped us to walk the path less abruptly.

## Annex

To save some tables that support our conclusions


