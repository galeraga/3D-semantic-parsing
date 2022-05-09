# http://www.open3d.org/docs/latest/introduction.html
# Pay attention to Open3D-Viewer App http://www.open3d.org/docs/latest/introduction.html#open3d-viewer-app
# and the Open3D-ML http://www.open3d.org/docs/latest/introduction.html#open3d-ml
# pip install open3d

# If import is not set this way, accessing o3d.ml.torch will throw an error:
# AttributeError: module 'open3d.ml' has no attribute 'torch'
from pathlib import Path
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
S3DIS_SUMMARY_FILE = "s3dis_summary.csv"

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


class S3DIS_Summarizer():
    """
    Class to get info from the S3DIS dataset
    """

    # Names of the cols are going to be saved in the CSV summary file
    S3DIS_summary_cols = ["Area", "Space", "Object", "Object Points", "Space Label", "Object Label"]

    def __init__(self, path_to_data, rebuild = False):
        """
        Inspect the dataset to get the following info:

            - Areas
            - Spaces
            - Objects
            - Points per object
            - Labels (for both areas and spaces)
        
        S3DIS dataset structure:

        Path_to_data\Area_N\space_X
                        \space_Y
                        \space_Z\Annotations\object_1
                        \space_Z\Annotations\object_2

        Input: 
            - Path to dataset

        Output: A CSV file containing the following columns:
            - Area
            - Space
            - Object
            - Points per object
            - Space Label
            - Object Label

        If rebuild is set to True, the summary file is generated again
        """
        
        self.path_to_data = path_to_data
        self.rebuild = rebuild

        # Do NOT process the info if the summary file already exists
        if self.check_summary_file() and (self.rebuild == False):
                msg = "Skipping summary file generation. The S3DIS summary file {} already exists in {}"
                print(msg.format(S3DIS_SUMMARY_FILE, self.path_to_data))
                return
        
        print("Generating summary file {} in {}".format(S3DIS_SUMMARY_FILE, self.path_to_data))

        # Every line of the S3DIS summary file will contain:
        # (area, space, object, points_per_object, space label, object label)
        summary_line = []
        
        # Keep only folders starting with Area_XXX
        areas = dict((folder, '') for folder in sorted(os.listdir(self.path_to_data)) if folder.startswith('Area'))

        # For every area folder, get the disjoint spaces included within it
        for area in areas:

            # os.path.join takes into account the concrete OS separator ("/", "\")        
            path_to_spaces = os.path.join(self.path_to_data, area)
            
            # Get spaces for each area, avoiding non disered files (".DStore", ...)
            spaces = sorted([space for space in os.listdir(path_to_spaces) 
                if not '.' in space])    
            
            # For every sapce, get the objects it contains
            for space in spaces:
                path_to_objects = os.path.join(path_to_spaces, space, "Annotations")
                
                # Get the space label
                # From hallway_1, hallway_2, take only "hallway"
                space_label = space.split("_")[0]
                
                # The file to be used will be the original of the S3DIS 
                # (not the_rgb_norm.txt), since rgb normalization is 
                # optional (only required to visualize data with Open3D)        
                objects = sorted([object for object in os.listdir(path_to_objects) 
                    if not PC_FILE_EXTENSION_RGB_NORM in object])    

                for object in objects:
                    # Get the object label
                    # From chair_1, chair_2, take only "chair"
                    object_label = object.split("_")[0]
    
                    # Avoid saving .DStore and other files
                    if "." not in object_label:
                        
                        # Get the number of points in the object
                        print("Getting points from file (object) {}_{}_{}".format(area, space, object))
                        with open(os.path.join(path_to_objects, object)) as f:
                            points_per_object = len(list(f))
                        
                        # Save all the traversal in the summary file:
                        # (Area, space, object, points per object, space label, object label)
                        summary_line.append((area, space, object, points_per_object, space_label, object_label))


        # Save the data into a CSV file
        summary_df = pd.DataFrame(summary_line)
        summary_df.columns = self.S3DIS_summary_cols
        summary_df.to_csv(os.path.join(PC_DATA_PATH, S3DIS_SUMMARY_FILE), index = False, sep = "\t")


    def check_summary_file(self):
        """
        Checks whether the summary file already exists
        """

        summary_existence = True if S3DIS_SUMMARY_FILE in os.listdir(self.path_to_data) else False

        return summary_existence
    

    def get_labels(self):
        """
        Get the labels from the S3DIS dataset folder structure

        Create dicts with the different spaces (conf rooms, hall ways,...)
        and objects (table, chairs,...) within an Area 
        
        Output:
            space_labels: A dict containing {0: space_0, 1: space_1, ... }
            object_labels: A dict containing {0: object_0, 1: object_1, ... }
        """
        
        if self.check_summary_file() == False:
            msg = "No S3DIS summary file {} found at {}."
            msg += "Summary file is going to be automatically generated"
            print(msg.format(S3DIS_SUMMARY_FILE, self.path_to_data))     
            self.__init__(self.path_to_data, rebuild = True)
        
        # Define the sets and dicts to be used 
        spaces_set = set()
        objects_set = set()
        space_labels = dict()
        object_labels = dict()

        # Open the CSV sumamry  file
        summary = os.path.join(self.path_to_data, S3DIS_SUMMARY_FILE)
                
        # Process each line in the summary file
        with open(summary) as f:
            for idx,line in enumerate(f):
                # Skip the first row (since it contain the header and no data)
                if idx != 0:
                    # Split the line, based on the tab separator
                    line = line.split("\t")       
                    # Add the space to the set             
                    spaces_set.add(line[4])                    
                    # Add the object to the set
                    # Remove the new line char at the end of every line for objects
                    objects_set.add(line[5].strip("\n"))

        # Create the idx-to-label dicts
        for idx, space in enumerate(spaces_set):
            space_labels[idx] = space
    
        for idx, object in enumerate(objects_set):
            object_labels[idx] = object

        return space_labels, object_labels

        
    def get_stats(self):
        """
        Get several statistics about the dataset
        """

        if self.check_summary_file() == False:
            msg = "No S3DIS summary file {} found at {}."
            msg += "Summary file is going to be automatically generated"
            print(msg.format(S3DIS_SUMMARY_FILE, self.path_to_data))     
            self.__init__(self.path_to_data, rebuild = True)
        
        # Open the CSV summary  file
        summary = os.path.join(self.path_to_data, S3DIS_SUMMARY_FILE)

        # Get the whole summary
        summary = pd.read_csv(summary, header =0, usecols = self.S3DIS_summary_cols, sep = "\t")
        
        # Get stat info about the Object Point col directly
        print("Points information:", summary.describe())

        # Total areas 
        areas = sorted(set(summary['Area']))
        print("Areas found:", areas)
        
        # Total spaces per area
        total_spaces = []
        for area in areas:
            # Returns a new dataframe containing only the proper area
            area_df = summary.loc[summary['Area'] == area]
            
            # For that area, get non-repeated spaces
            spaces_per_area = len(sorted(set(area_df["Space"])))
            print("Total spaces in area {}: {}".format(
                area, spaces_per_area))
            total_spaces.append(spaces_per_area)
        print("Total spaces: ", sum(total_spaces))
       
        # Total objects
        # Minus one to remove the header row
        print("Total objects: ", len(summary.index)-1)

        # Total points
        # Minus one to remove the header row
        object_points_df = summary["Object Points"]
        print("Total points: ", object_points_df.sum())
        print("Max points per object: ", object_points_df.max())
        print("Min points per object: ", object_points_df.min())
        #TODO: quantiles and percentiles of points

        
        # TODO: Total objects per space
        ...

        # TODO: Points per area
        ...

        # TODO: Points per space
        ...

        # TODO: Points per kind of object
        ...
        

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


if __name__ == "__main__":
    
    # Two minor issues when working with S3DIS dataset:
    # - Open3D does NOT support TXT file extension, so we have to specify 
    #   the xyzrgb format (check supported file extensions here: 
    #   http://www.open3d.org/docs/latest/tutorial/Basic/file_io.html) 
    # - When working with xyzrgb format, each line contains [x, y, z, r, g, b], 
    #   where r, g, b are in floats of range [0, 1]
    #   So we need to normalize the RGB values from the S3DIS dataset in order 
    #   to allow Open3D to display them

    # Create the summary file that will contain important info about the dataset
    summary = S3DIS_Summarizer(PC_DATA_PATH)
    
    # Get the labels 
    space_labels, object_labels = summary.get_labels()
    
    # Get statistical info
    summary.get_stats()

    # TODO: To be removed, since all data is based now in the summary file
    # Get a dict of areas and spaces
    areas_and_spaces = get_spaces(PC_DATA_PATH)

    # TODO: Rebuild the normalization to be based on the sumamry file,
    # not in the traversal of directories
    # Normalize RGB in all spaces
    # RGB_normalization(areas_and_spaces)

    # To quickly test o3d
    pcd = o3d.io.read_point_cloud(
        os.path.join(PC_DATA_PATH, TEST_PC + PC_FILE_EXTENSION_RGB_NORM),
        format='xyzrgb')
    print(pcd)
    # o3d.visualization.draw_geometries([pcd])
