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

This project'll be focus on implementing only **object classification** and **scene semantic parsing**. As a dataset, the **S3DIS dataset** is going to be used, where every point includes both its spatial coordinates and its color info (xyzrgb).

## Main goals
This is section1 

## The dataset
The **3D Semantic Parsing of Large-Scale Indoor Spaces (S3DIS)** dataset is going to be used in order to work with the PointNet architecture. The dataset contains point clouds from **3 different buildings**, distributed in **6 areas**:

- Building 1: Area_1, Area_3, Area_6 
- Building 2: Area_2, Area_4 
- Building 3: Area_5 

There're **272 rooms** (or spaces) dsitributed among all areas, and every room can have up to **14 different objects**. These object can be classified either as **movable** (boards, bookcases, chairs, tables and sofas) or **structural** (ceilings, doors, floors, walls, beams, columns, windows and stairs). If an object doesn't belong to any of the previous catagories, it's classied as clutter. 


The folder structure of the S3DIS dataset is the following one:
```
Area_N\Space_X\space_x.txt (the non-annotated file with the point cloud for this space)
|-------------\Annotations\object_1.txt (the file with the point cloud for object_1 that can be found in Space_X)
.
.
.
|-------------------------\object_Y.txt (the file with the point cloud for object_Y that can be found in Space_X)

More info about the S3DIS dataset can be found at: http://buildingparser.stanford.edu/dataset.html 

From this original S3DIS dataset, two custom datasets have been created
```


### Subsection 1
This is subsection 1.1 

#### Subsection 1.1.1
This is subsection 1.1.1 

## Repository Structure
This is section1 

## Related Work

## Contributors
This is section1 

## Acknowledgments
This is section1 


