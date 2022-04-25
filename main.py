# http://www.open3d.org/docs/latest/introduction.html
# Pay attention to Open3D-Viewer App http://www.open3d.org/docs/latest/introduction.html#open3d-viewer-app
# and the Open3D-ML http://www.open3d.org/docs/latest/introduction.html#open3d-ml
# pip install open3d
from fileinput import filename
from multiprocessing.sharedctypes import Value
import open3d as o3d
import torch
import os
import logging

# Set the sample path HERE:
POINT_CLOUD_DATA_PATH = "/Users/jgalera/datasets/S3DIS"
TEST_PC = "/Area_1/office_1/office_1.txt"
PC_FILE_EXTENSION = ".txt"
PC_FILE_EXTENSION_RGB_NORM = "_rgb_norm.txt"
LOG_FILE = "conversion.log"

# Define the logging settings
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename = os.path.join(POINT_CLOUD_DATA_PATH, LOG_FILE),
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
    if tgt_file in os.listdir(file_path):
        print("Skipping RGB normalization of file ", f)
        return
    else:
        print("RGB normalization of file ", f)
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
                msg = "Unable to procees file " + f
                print(msg)
                logging.warning(msg)
                
    
    return tgt_file


def normalize_RGB(spaces):
    """
    Normalize RGB in all disjoint spaces in order to let o3d display them
    """

    for area in spaces:
        for folder in spaces[area]:        
            normalize_RGB_single_file(os.path.join(POINT_CLOUD_DATA_PATH, area, folder, folder) + PC_FILE_EXTENSION)


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

    # Get the areas and spaces to define datasets and dataloaders
    avaialable_spaces = get_spaces(POINT_CLOUD_DATA_PATH)

    # Normalize RGB in all spaces
    normalize_RGB(avaialable_spaces)

    # To quickly test o3d
    pcd_RGB_normalized = normalize_RGB_single_file(POINT_CLOUD_DATA_PATH + TEST_PC)
    pcd = o3d.io.read_point_cloud(pcd_RGB_normalized, format='xyzrgb')
    print(pcd)
    o3d.visualization.draw_geometries([pcd])
