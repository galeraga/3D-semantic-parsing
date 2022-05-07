# http://www.open3d.org/docs/latest/introduction.html
# Pay attention to Open3D-Viewer App http://www.open3d.org/docs/latest/introduction.html#open3d-viewer-app
# and the Open3D-ML http://www.open3d.org/docs/latest/introduction.html#open3d-ml
# pip install open3d

# If import is not set this way, accessing o3d.ml.torch will throw an error:
# AttributeError: module 'open3d.ml' has no attribute 'torch'
import open3d.ml.torch as ml3d

import open3d as o3d
import torch
import os
import sys
import logging
import pandas as pd


# Set the sample path HERE:
PC_DATA_PATH = "/Users/jgalera/datasets/S3DIS/byhand"
# Select what do yo want to visualize: an space or and object
# Uncomment the following line if yoy want to visualize an space
#TEST_PC = "Area_1/office_1/office_1"
# Uncomment the following line if yoy want to visualize an object
TEST_PC = "Area_1/office_1/Annotations/table_1"

PC_FILE_EXTENSION = ".txt"
PC_FILE_EXTENSION_RGB_NORM = "_rgb_norm.txt"
LOG_FILE = "conversion.log"
PC_SPACE_SUMMARY_FILE = "summary_pc_per_space.csv"
PC_OBJECT_SUMMARY_FILE = "summary_pc_per_object.csv"

# Define the logging settings
# Logging is Python-version sensitive
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# For Python < 3.9 (minor version: 9) 
# encoding argument can't be used
if sys.version_info[1] < 9:
    logging.basicConfig(filename = os.path.join(PC_DATA_PATH, LOG_FILE),
        level=logging.WARNING,
        format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename = os.path.join(PC_DATA_PATH, LOG_FILE),
        encoding='utf-8', 
        level=logging.WARNING,
        format='%(asctime)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_RGB_single_file(f):
    """
    Takes the input file and calculates the RGB normalization
    for a single point cloud file
    """

    # Keep the original dataset file intact and create 
    # a new file with normalized RGB values 
    file_path, file_name = os.path.split(f)   
    tgt_file = file_name.split('.')[0] + PC_FILE_EXTENSION_RGB_NORM
     
    # Skip the process if the file has been already normalized
    if (tgt_file in os.listdir(file_path)) or (PC_FILE_EXTENSION_RGB_NORM in file_name):
        print("...skipped (already normalized)")
        return
    else:
        tgt_file = os.path.join(file_path, tgt_file)

    normalized = ''
    with open(f) as src:
        with open(tgt_file, "w") as tgt:
            try:
                for l in src:
                    # Convert the str to list for easier manipulation
                    x, y, z, r, g, b = l.split()
                    r = float(r)/255
                    g = float(g)/255
                    b = float(b)/255

                    # Back to str again
                    normalized += ' '.join([str(x), str(y), str(z), 
                        '{:.8s}'.format(str(r)), 
                        '{:.8s}'.format(str(g)), 
                        '{:.8s}'.format(str(b)), 
                        '\n'])        
                
                tgt.write(normalized)

            except ValueError:
                msg1 = " -> unable to procees file %s " % src.name
                msg2 = msg1 + "(check log at %s)" % os.path.join(PC_DATA_PATH, LOG_FILE)
                print(msg2)
                logging.warning(msg1)
            
            else:
                print("...done")


def RGB_normalization(areas):
    """
    Normalize RGB in all disjoint spaces in order to let o3d display them
    """
    # Let's gather the total number of spaces to process    
    total_areas = len(areas)
    total_spaces = 0
    for space in areas:
        total_spaces += len(areas[space])

    # Let's process each space
    total_processed = 0
    for idx, (area, folders) in enumerate(sorted(areas.items())):
        for folder in sorted(folders):    
            total_processed += 1         
            print("Processing RGB normalization in {} ({}/{})| file {} ({}/{})".format(
                area, (idx+1), total_areas, folder, total_processed, total_spaces), 
                end = " ")
            path_to_space = os.path.join(PC_DATA_PATH, area, folder)
            normalize_RGB_single_file(os.path.join(path_to_space, folder) + PC_FILE_EXTENSION)

            # Let's also process the annotations
            path_to_annotations = os.path.join(path_to_space,"Annotations")
            for file in os.listdir(path_to_annotations):
                print("\tProcessing RGB normalization in file: ", file, end = " ")
                normalize_RGB_single_file(os.path.join(path_to_annotations, file))




def get_spaces(path_to_data):
    """
    Inspect the dataset location to determine the amount of available 
    areas and spaces (offices, hallways, etc) 

    Path_to_data\Area_N\office_X
                       \office_Y
                       \office_Z

    Input: Path to dataset
    Output: A dict with 
        - key: Area_N
        - values: a list of included disjoint spaces per Area
    """
    
    # Keep only folders starting with Area_XXX
    areas = dict((folder, '') for folder in os.listdir(path_to_data) if folder.startswith('Area'))
    
    # For every area folder, get the disjoint spaces included within it
    # Removing any file that contains '.' (e.g., .DStore, alignment.txt)
    # os.path.join takes into account the concrete OS separator ("/", "\")
    for area in areas:
        areas[area] = sorted([subfolder for subfolder in os.listdir(os.path.join(path_to_data, area)) 
            if not '.' in subfolder])

    return areas
    
def get_labels(areas):
    """
    Create dicts with the different spaces ((conf rooms, hall ways,...)
    and objects (table, chairs,...) within an Area 
    
    Needed to label obejcts and spaces

    Input:
        A dict with 
            - key: Area_N
            - values: a list of included disjoint spaces per Area
    Output:
        dict_spaces: A dict containing {0: space_0, 1: space_1, ... }
        dict_objects: A dict containing {0: object_0, 1: object_1, ... }
    """
    
    s3dis_spaces = set()
    s3dis_objects = set()
    dict_spaces = dict()
    dict_objects = dict()

    for area, folders in sorted(areas.items()):
        for folder in folders:
            # Get space names
            # From hallway_1, hallway_2, take only "hallway"
            space_name = folder.split("_")[0]
            s3dis_spaces.add(space_name)

            # Get object names
            path_to_objects = os.path.join(
                PC_DATA_PATH, 
                area, 
                folder,
                "Annotations"
            )

            for file in os.listdir(path_to_objects):
                object_name = file.split("_")[0]
                # Avoid saving .DStore and other files
                if "." not in object_name:
                    s3dis_objects.add(object_name)

    for idx, space in enumerate(s3dis_spaces):
        dict_spaces[idx] = space
    
    for idx, object in enumerate(s3dis_objects):
        dict_objects[idx] = object

    return dict_spaces, dict_objects


def get_points(areas):
    """
    Calculate the number of total points in every space (office_1, storage_1,...)
    and every object (table, chair, ...)
    
    Input: 
        A dict with 
            - key: Area_N
            - values: a list of included disjoint spaces per Area
    Output:
        - A CSV file containing the summary of total points per space
        - A CSV file containing the summary of total points per object
    """

    skip_space_processing = False
    skip_object_processing = False

    if PC_SPACE_SUMMARY_FILE in os.listdir(PC_DATA_PATH):
                print("Skipping point gathering for spaces. File {} already exists in {}".format(PC_SPACE_SUMMARY_FILE, PC_DATA_PATH))
                skip_space_processing = True
                
    if PC_OBJECT_SUMMARY_FILE in os.listdir(PC_DATA_PATH):
                print("Skipping point gathering for objects. File {} already exists in {}".format(PC_SPACE_SUMMARY_FILE, PC_DATA_PATH))
                skip_object_processing = True
            
    space_points = []
    object_points = []
    
    for area, folders in sorted(areas.items()):
        for folder in folders:      
            # Get the total points in every space
            path_to_space = os.path.join(
                PC_DATA_PATH, 
                area, 
                folder, 
                folder + PC_FILE_EXTENSION_RGB_NORM
            )
            if skip_space_processing == False:
                print("Getting points from file (space) {}_{}".format(area, folder))
                with open(path_to_space) as f:
                    space_points.append((area, folder, len(list(f))))
        
            # Get the total points in every object object within space
            path_to_objects = os.path.join(
                PC_DATA_PATH, 
                area, 
                folder,
                "Annotations"
            )
            
            if skip_object_processing == False:
                for file in os.listdir(path_to_objects):
                    # Let's process only the RGB normalized
                    if PC_FILE_EXTENSION_RGB_NORM in file:
                        print("Getting points from file (object) {}_{}_{}".format(area, folder, file))
                        with open(os.path.join(path_to_objects, file)) as f:
                            object_points.append((area, folder, file, len(list(f))))
                       
    # Saving ther results into a CSV to avoid processing this data again
    spaces_df = pd.DataFrame(space_points)
    spaces_df.to_csv(os.path.join(PC_DATA_PATH, PC_SPACE_SUMMARY_FILE), index = False, sep = " ")

    objects_df = pd.DataFrame(object_points)
    objects_df.to_csv(os.path.join(PC_DATA_PATH, PC_OBJECT_SUMMARY_FILE), index = False, sep = " ")


if __name__ == "__main__":
    
    
    # Two minor issues when working with S3DIS dataset:
    # - Open3D does NOT support TXT file extension, so we have to specify 
    #   the xyzrgb format (check supported file extensions here: 
    #   http://www.open3d.org/docs/latest/tutorial/Basic/file_io.html) 
    # - When working with xyzrgb format, each line contains [x, y, z, r, g, b], 
    #   where r, g, b are in floats of range [0, 1]
    #   So we need to normalize the RGB values from the S3DIS dataset in order 
    #   to allow Open3D to display them

    # Get a dict of areas and spaces
    areas_and_spaces = get_spaces(PC_DATA_PATH)

    # Normalize RGB in all spaces
    RGB_normalization(areas_and_spaces)

    # Gather labels from folder traversal for both spaces and objects
    spaces_dict, objects_dict = get_labels(areas_and_spaces)
    print("Labels for spaces: ", spaces_dict)
    print("Labels for objects: ", objects_dict)

    # Get the points in both spaces and objects
    get_points(areas_and_spaces)


    # To quickly test o3d
    pcd = o3d.io.read_point_cloud(
        os.path.join(PC_DATA_PATH, TEST_PC + PC_FILE_EXTENSION_RGB_NORM),
        format='xyzrgb')
    print(pcd)
    # o3d.visualization.draw_geometries([pcd])



    # The following lines have to be removed. They're test with the datset 
    # included in the o3d.ml library
    """"
    # http://www.open3d.org/docs/release/python_api/open3d.ml.torch.datasets.S3DIS.html
    ml3d_dataset = ml3d.datasets.S3DIS(
        dataset_path = POINT_CLOUD_DATA_PATH,
        name = "S3DIS",
        # {segmentation, detection}
        task = "segmentation",
        test_area_idx = 2,
        test_result_folder = os.path.join(POINT_CLOUD_DATA_PATH, "results")
    )

    
    print("ml3d_dataset: ", ml3d_dataset)
    print("Labels for classification: ", ml3d_dataset.get_label_to_names())

    # get the 'all' split that combines training, validation and test set
    ml3d_all = ml3d_dataset.get_split('all')
    ml3d_training = ml3d_dataset.get_split('training')
    ml3d_test = ml3d_dataset.get_split('test')
    ml3d_val = ml3d_dataset.get_split('validation')
    
    my_dsets = [("all", ml3d_all), ("training", ml3d_training), 
        ("test", ml3d_test), ("validation", ml3d_val)]

    for dset in my_dsets:
        print("{} dataset -> Lenght {} (id: {})".format(dset[0], len(dset[1]), dset))  
    
    # Attr:  {'idx': 1, 'name': 'Area_1_conferenceRoom_1', 'path': '/Users/jgalera/datasets/S3DIS/byhand/original_pkl/Area_1_conferenceRoom_1.pkl', 'split': 'all'}
    for name, dset in my_dsets:
        print("Processing ATTRIBUTES from dataset {}".format(name))
        for i in range(len(dset)):
            print("Attribute: ", dset.get_attr(i))


    # https://github.com/isl-org/Open3D-ML/blob/master/ml3d/datasets/s3dis.py
    # get_data -> data = {'point': points, 'feat': feat, 'label': labels, 'bounding_boxes': bboxes
    for name, dset in my_dsets:
        print("Processing DATA from dataset {}: ".format(name))
        for i in range(len(dset)):
            print("\tPoint shape: ", dset.get_data(i)['point'].shape)
            print("\tFeature shape: ", dset.get_data(i)['feat'].shape)
            print("\tLabel shape: ", dset.get_data(i)['label'].shape)
            print("\tBounding Box: ", dset.get_data(i)['bounding_boxes'])



    # show the first 100 frames using the visualizer
    # vis = ml3d.vis.Visualizer()
    # vis.visualize_dataset(ml3d_dataset, 'all', indices=range(3))

    train_dataloader = ml3d.dataloaders.TorchDataloader(
        dataset=ml3d_dataset.get_split('training'),
        preprocess = None,
        transform = None,
        steps_per_epoch = 64)
    
    print("Train dataloader", train_dataloader)
    # Returns the steps_per_epoch arg
    #print("Len dataloader: ", len(train_dataloader))
    #print(train_dataloader.__getitem__(64)['data'])
    #print(train_dataloader.__getitem__(64)['attr'])

        
    #for step in range(len(ml3d_training)):  # one pointcloud per step
    #    ml3d_training.get_data(step)['point']
    #    ml3d_training.get_data(step)['point']
    
"""