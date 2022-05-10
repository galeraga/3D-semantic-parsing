
from settings import *

class S3DIS_Summarizer():
    """
    Class to get info from the S3DIS dataset
    """

    # Names of the cols are going to be saved in the CSV summary file
    # after folder traversal
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
                        
                        # Save all the traversal info in the summary file:
                        # (Area, space, object, points per object, space label, object label)
                        summary_line.append((area, space, object, points_per_object, space_label, object_label))


        # Save the data into the CSV summary file
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

        # Open the CSV summary file
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
        
        # Open the CSV summary file
        summary = os.path.join(self.path_to_data, S3DIS_SUMMARY_FILE)

        # Get the whole summary
        summary = pd.read_csv(summary, header =0, usecols = self.S3DIS_summary_cols, sep = "\t")
        
        # Get stat info about the Object Point col directly
        print("Statistical info about the points:", summary.describe())

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
        print("Mean points per object: ", object_points_df.mean())
        print("Median points per object: ", object_points_df.median())
        print("Quantile 90%: ", object_points_df.quantile(0.90))
        #TODO: quantiles and percentiles of points

        
        # TODO: Total objects per space
        ...

        # TODO: Points per area
        ...

        # TODO: Points per space
        ...

        # TODO: Points per kind of object
        ...
