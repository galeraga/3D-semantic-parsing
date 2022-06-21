# 3D-semantic-parsing
Repo to host the UPC AIDL spring 2022 post-graduate project

## Abstract
A point cloud is a type of 3D geometric data structure, based on unordered set of vectors.
Each point is a vector of its (z, y, z) coordinate plus extra feature channels such as color, normal, etc.

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

More **advanced statistics** can be found after a more detailed analysis:

<img width="1043" alt="image" src="https://user-images.githubusercontent.com/76537012/174793841-01903c11-10c2-425c-a6d5-c0a25391ccc0.png">

The original **folder structure** of the S3DIS dataset is the following one:
```
Area_N\
  |- space_X\
      |- space_x.txt (the non-annotated file with the point cloud for this space)
      |- Annotations\
          |- object_1.txt (the file with the point cloud for object_1 that can be found in Space_X)
          |- ...
          |- object_Y.txt (the file with the point cloud for object_Y that can be found in Space_X)
```  

- ` object_Y.txt` is a file containing the point cloud of this particular object that belongs to Space_X (e.g., objects *chair_1.txt, chair_2.txt* or *table_1.txt* from an space/room called *office_1*). This point cloud file has 6 columns (non-normalized XYXRGB values).
- ` space_x.txt` is a non-annotated point cloud file containing the sum of of all the point cloud object files (object_1, object_2, ...) located within the `Annotations` folder (e.g., the space file called *Area_1\office_1\office_1.txt* contains the sum of the object files *chair_1.txt, chair_2.txt* and the rest of all the object files located inside the `Annotations` directory). As a consequence, the space/room point cloud file has only 6 columns too (non-normalized XYXRGB values).

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

This is the dataset used for **semantic segmentation**. Since semantic segmentation needs every point in the cloud to be labeled, a new file is generated for every space/room with the suitable object labels. To do so, all files located in the `Annotations` folder are concatenated (along with the proper label) to create a single file per space/room with all the object points that belong to this space already labaled. This file is called `space_x_annotated.txt` (e.g., *office_1_annotated.txt*) and contains 7 cols (instead of 6): XYZRGB+*Object ID*. 

So the S3DISDataset4Segmentation will use the `space_x_annotated.txt` to get both the **input data** and **labels**

### The final folder structure 

Taking into account the previous information, the final folder structure for the model implemenation is the following one:
```
s3dis_summary.csv (the ground truth file)
Area_N\
  |- space_X\
      |- space_x.txt (the non-annotated file with the point cloud for this space. It only contains 6 cols per row: XYZRGB)
      |- space_x_annotated.txt (the annotated file with the point cloud for this space. It contains 7 cols per row: XYZRGB+*Object ID*)
      |- Annotations\
          |- object_1.txt (the file with the point cloud for object_1 that can be found in Space_X. It contains 6 cols per row: XYZRGB)
          |- ...
          |- object_Y.txt (the file with the point cloud for object_Y that can be found in Space_X. It contains 6 cols per row: XYZRGB)
```  


## The model

###TransformationNet (picture showing the T-Net)###

###BasePointNet###

<img width="962" alt="image" src="https://user-images.githubusercontent.com/76537012/174834124-c25dbcfe-2c46-4616-88dd-af417c0975ee.png">


###ClassificationPointNet###


###SegmentationPointNet###

## Related Work

## Contributors
This is section1 

## Acknowledgments
This is section1 


