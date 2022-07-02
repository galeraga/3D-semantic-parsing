
#from tensorboard import summary
from settings import *

#import summarizer

#------------------------------------------------------------------------------
# Helper classes and methods, shared among classification and segmentation goals
#------------------------------------------------------------------------------

def get_summary_file():
    """
    Returns a Panda Dataframe for the summary file
    """
    summary_df = pd.read_csv(
                os.path.join(eparams['pc_data_path'], eparams['s3dis_summary_file']), 
                header = None, 
                skiprows=1,
                sep = "\t"
            )
    return summary_df


class PointSampler():
    """
    Utility class to downsampling/upsamplng a point cloud
    in order to make all point clouds the same size
    """
    def __init__(self, point_cloud, max_points):
        """
        Args:
            point_cloud: torch file
            max_points: The max amount of points the point cloud must have
        """
        self.point_cloud = point_cloud
        self.max_points = max_points

    def sample(self):

        # Check if the point cloud is already a tensor
        if not torch.is_tensor(self.point_cloud):
            self.point_cloud = torch.tensor(self.point_cloud)

        # Since point cloud tensors will have different amount of points/row,
        # we need to set a common size for them all
        # Torch Dataloaders expects each tensor to be equal size
        if(len(self.point_cloud) > self.max_points):   
            # Sample points 
            # Shuffle the index (torch.randperm(4) -> tensor([2, 1, 0, 3]))
            idxs = torch.randperm(len(self.point_cloud))[:self.max_points]
            self.point_cloud = self.point_cloud[idxs]
        
        else:
            # Duplicate points
            point_cloud_len = len(self.point_cloud)
            padding_points = self.max_points - point_cloud_len     
            for _ in range(padding_points):     
                duplicated_point_cloud_points = torch.unsqueeze(self.point_cloud[random.randint(0, point_cloud_len - 1)], dim = 0)
                self.point_cloud = torch.cat((self.point_cloud, duplicated_point_cloud_points), dim = 0)
        
        return self.point_cloud

class AdaptNumClasses():
    """
    Adapts the point_labels tensor to have only the proper classes:
    
    movable objects: 5 + clutter
    structural objects: 8 + clutter
    all objects: 13 + clutter

    movable_objects = set("board", "bookcase", "chair", "table", "sofa", "clutter")
    structural_objects = set("ceiling", "door", "floor", "wall", "beam", "column", "window", "stairs", "clutter")


    """
    def __init__(self, point_labels, all_dicts):
        """
        Args:
            point_labels: torch column vector
        """
        self.point_labels = point_labels
        self.all_dicts = all_dicts
        # Get the vector type in order to keep them when replacing
        
    def adapt(self):
        """
        Assign a clutter label (object_ID: 1) to any point not intended to be
        trained and remap the object IDs to values expected by the loss function

        From the summary file, these are the available dicts:
        'all': {'ceiling': 0, 'clutter': 1, 'door': 2, 'floor': 3, 'wall': 4, 'beam': 5, 'board': 6, 'bookcase': 7, 'chair': 8, 'table': 9, 'column': 10, 'sofa': 11, 'window': 12, 'stairs': 13}, 
        'movable': {'clutter': 0, 'board': 1, 'bookcase': 2, 'chair': 3, 'table': 4, 'sofa': 5}, 
        'structural': {'ceiling': 0, 'clutter': 1, 'door': 2, 'floor': 3, 'wall': 4, 'beam': 5, 'column': 6, 'window': 7, 'stairs': 8}}
        
        The 'all' dict is computed during summary file creation, whereas 
        'movable' and 'structural' are created afterwards in method get_labels() 
        from summarizer.py

        Example on how remapping works for a chair:
        - From summary_file, chair has object_ID = 8
        - If working only with movable objects, the 'movable' dict has valur 3 for chair
        - So the new object_ID for chair when working with movable objects must 
          be 3 for the loss function to have all values between [0, len(movable)].
          A value of 8 is not supported by the loss function when working with 
          movable objects only. All values must be in the range from 0 - len(dict)
        """
        
        target_objects = ''.join(args.objects)       
        from_dict = self.all_dicts["all"]
        to_dict = self.all_dicts[target_objects]
        
        # classification labels are ints (no lists, no tensor)
        # so we convert the int to list for the remapping loop to work
        # with both int classification labels and tensors segmentation labels
        if "classification" in args.goal:
            self.point_labels = list([self.point_labels])
        
        # Remapping is only needed when working with movable or structural objects
        if target_objects != "all":
            for i in range(len(self.point_labels)):            
                    # Get the textual label of the point from the "old/from" dict
                    textual_label = ''.join([k for k,v in from_dict.items() if v == self.point_labels[i]])
                    
                    # If the object is not defined in the "new/to" dict, flag it as clutter
                    if textual_label not in to_dict.keys():
                        self.point_labels[i] = to_dict["clutter"]
                    
                    # Remap/translate the rest of the objects
                    else:
                        self.point_labels[i] = to_dict[textual_label]

         
        return self.point_labels

#------------------------------------------------------------------------------
# Datasets for Classification
#------------------------------------------------------------------------------

class S3DISDataset4ClassificationBase(torch.utils.data.Dataset):
    """
    Python base class for creating the S3DIS datasets for classification goals.

    The datasets are created from the summary_file.csv, using Pandas DataFrames.

    Dataset elements are chosen based on the area (proper_area list arg, defined 
    in settings.py) they belong to:

    - Training dataset: Areas 1, 2, 3 and 4
    - Validation dataset: Area 5
    - Test dataset: Area 6
   
    Training dataset: 
            0          1        2  3               4       5        6  7     8
        0     Area_1       WC_1       WC  0   ceiling_1.txt  179217  ceiling  0  Good
        1     Area_1       WC_1       WC  0   ceiling_2.txt   12822  ceiling  0  Good
        2     Area_1       WC_1       WC  0   clutter_1.txt    8147  clutter  1  Good
        3     Area_1       WC_1       WC  0  clutter_10.txt   23023  clutter  1  Good
        4     Area_1       WC_1       WC  0  clutter_11.txt    7559  clutter  1  Good
        ...      ...        ...      ... ..             ...     ...      ... ..   ...
        5808  Area_4  storage_4  storage  7      door_2.txt   10611     door  2  Good
        5809  Area_4  storage_4  storage  7     floor_1.txt   32590    floor  3  Good
        5810  Area_4  storage_4  storage  7      wall_1.txt    6733     wall  4  Good
        5811  Area_4  storage_4  storage  7      wall_2.txt   30409     wall  4  Good
        5812  Area_4  storage_4  storage  7      wall_3.txt   19603     wall  4  Good

        [5813 rows x 9 columns]
    
    Validation dataset:
                0          1        2  3               4       5        6  7     8
        0     Area_5       WC_1       WC  0   ceiling_1.txt  182136  ceiling  0  Good
        1     Area_5       WC_1       WC  0   clutter_1.txt    6431  clutter  1  Good
        2     Area_5       WC_1       WC  0  clutter_10.txt    2682  clutter  1  Good
        3     Area_5       WC_1       WC  0  clutter_11.txt    3224  clutter  1  Good
        4     Area_5       WC_1       WC  0  clutter_12.txt    2315  clutter  1  Good
        ...      ...        ...      ... ..             ...     ...      ... ..   ...
        2341  Area_5  storage_4  storage  7     table_2.txt    6469    table  9  Good
        2342  Area_5  storage_4  storage  7      wall_1.txt   42265     wall  4  Good
        2343  Area_5  storage_4  storage  7      wall_2.txt   90417     wall  4  Good
        2344  Area_5  storage_4  storage  7      wall_3.txt   82434     wall  4  Good
        2345  Area_5  storage_4  storage  7      wall_4.txt  145942     wall  4  Good

        [2346 rows x 9 columns]
    
    Test dataset
                0                 1               2  3              4       5        6  7     8
        0     Area_6  conferenceRoom_1  conferenceRoom  1     beam_1.txt   42884     beam  5  Good
        1     Area_6  conferenceRoom_1  conferenceRoom  1    board_1.txt   25892    board  6  Good
        2     Area_6  conferenceRoom_1  conferenceRoom  1    board_2.txt   19996    board  6  Good
        3     Area_6  conferenceRoom_1  conferenceRoom  1  ceiling_1.txt  208995  ceiling  0  Good
        4     Area_6  conferenceRoom_1  conferenceRoom  1    chair_1.txt    6548    chair  8  Good
        ...      ...               ...             ... ..            ...     ...      ... ..   ...
        1669  Area_6          pantry_1          pantry  5    table_2.txt   22381    table  9  Good
        1670  Area_6          pantry_1          pantry  5     wall_1.txt   56718     wall  4  Good
        1671  Area_6          pantry_1          pantry  5     wall_2.txt   47069     wall  4  Good
        1672  Area_6          pantry_1          pantry  5     wall_3.txt   13703     wall  4  Good
        1673  Area_6          pantry_1          pantry  5     wall_4.txt   27985     wall  4  Good

        [1674 rows x 9 columns]

    """
    
    def __init__(self, root_dir, all_objects_dict,  transform = None, proper_area = None):
        """
        Args:
            root_dir (string): Directory with all the point cluouds 
                and summary file
            transform (callable, optional): Optional transform 
                to be applied on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.all_objects_dict = all_objects_dict
        self.proper_area = proper_area
        
        # The ground truth file    
        # summary_df[0] is the "Area" col
        self.summary_df = get_summary_file()
        self.proper_df = self.summary_df[self.summary_df[0].isin(self.proper_area)].reset_index(drop = True)
    
    def __len__(self):
        """
        Returns only the objects marked as "Good" in the summary file
        to avoid errors while processing data
        """
        healthy_objects = self.proper_df[self.proper_df[8] == "Good"]
        return len(healthy_objects) -1 

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the summary line (by idx) which contains the proper 
        # object point cloud info
        summary_line = self.proper_df.iloc[idx]

        # Get the point cloud info from the summary line
        area = summary_line[0]
        space = summary_line[1]
        space_label = summary_line[2]
        space_label_id = summary_line[3]
        obj_file = summary_line[4]
        total_obj_points = summary_line[5]
        obj_label = summary_line[6]
        obj_label_id = summary_line[7]
        obj_health = summary_line[8]
          
        # Fetch the object point cloud, if object health is good:
        if obj_health == "Good":
            path_to_obj = os.path.join(self.root_dir, area, space, "Annotations", obj_file)
            
            # Element order is ignored when data is retrieved by cols in Pandas
            # so we need to define the order of the cols
            cols_to_get = [col for col in range (hparams['dimensions_per_object'])]
            obj_df = pd.read_csv(
                path_to_obj, 
                sep = " ", 
                dtype = np.float32, 
                header = None,
                usecols = cols_to_get 
                )[cols_to_get]

            # Convert the Pandas DataFrame to tensor
            obj = torch.tensor(obj_df.values).to(hparams["device"])

            # Make all the point clouds equal size
            obj = PointSampler(obj, hparams['num_points_per_object']).sample()

            # Adapt the labels to num classes
            # By default, points in the sliding windows have all labels (14)
            # We must adapt the point_labels tensor to have only the proper classes:
            # movable objects: 5 + clutter
            # structural objects: 8 + clutter
            # all objects: 13 + clutter
            
            labels = AdaptNumClasses(obj_label_id, self.all_objects_dict).adapt()
            labels = torch.tensor(labels, dtype = torch.float)    

            return obj, labels
    
    def __str__(self) -> str:
        
        msg_list = []
        msg_list.append(80 * "-")
        msg_list.append("S3DIS DATASET INFORMATION ({})".format(self.__class__.__name__))
        msg_list.append(80 * "-")
        msg_list.append("Summary file: {}".format(eparams['s3dis_summary_file']))
        msg_list.append("Data source folder: {} ".format(self.root_dir))
        msg_list.append("Chosen areas: {} ".format(self.proper_area))
        msg_list.append("Dataset elements:")
        msg_list.append(str(self.proper_df))
        msg = '\n'.join(msg_list)
        msg += "\n"

        return str(msg)


class S3DISDataset4ClassificationTrain(S3DISDataset4ClassificationBase):
    """
    Augmented S3DISDataset4ClassificationBase class to create the dataset used
    for training
    """
    def __init__(self, root_dir, all_objects_dict, transform = None, proper_area = None):
        S3DISDataset4ClassificationBase.__init__(self, root_dir, all_objects_dict, transform = None, proper_area = training_areas)

class S3DISDataset4ClassificationVal(S3DISDataset4ClassificationBase):
    """
    Augmented S3DISDataset4ClassificationBase class to create the dataset used
    for validation
    """
    def __init__(self, root_dir, all_objects_dict, transform = None, proper_area = None):
        S3DISDataset4ClassificationBase.__init__(self, root_dir, all_objects_dict, transform = None, proper_area = val_areas)

class S3DISDataset4ClassificationTest(S3DISDataset4ClassificationBase):
    """
    Augmented S3DISDataset4ClassificationBase class to create the dataset used
    for testing
    """
    def __init__(self, root_dir, all_objects_dict, transform = None, proper_area = None):
        S3DISDataset4ClassificationBase.__init__(self, root_dir, all_objects_dict, transform = None, proper_area = test_areas)

#------------------------------------------------------------------------------
# Datasets for Segmentation
#------------------------------------------------------------------------------

class S3DISDataset4SegmentationBase(torch.utils.data.Dataset):
    """
    Python base class for creating the S3DIS datasets for segmentation goals.

    The datasets are created from the contents of the sliding windows folder

    Dataset elements are chosen based on the area (proper_area list arg, defined 
    in settings.py) they belong to:

    - Training dataset: Areas 1, 2, 3 and 4
    - Validation dataset: Area 5
    - Test dataset: Area 6

    From the S3DIS_Summarizer.create_sliding_windows() method, 
    there's a folder containing all the pre-processed 
    sliding block for all the spaces/rooms in the dataset. 
    
    Naming guideline:

    Area_N
    sliding_windows
    ├── w_X_d_Y_h_Z_o_T
        ├── Area_N_Space_J_winK.pt

    where:    
    w_X: width of the sliding window
    d_Y: depth of the sliding window
    h_Z: height of the sliding window
    o_T: overlapping of consecutives sliding window
    winK: sequential ID of the sliding window
    """
    
    def __init__(self, root_dir, all_objects_dict, transform = None, proper_area = None, subset = None):
        """
        Args:
            root_dir (string): Directory with all the pre-processed 
                sliding windows 
            all_objects_dict: dict containing the mapping object_ID <-> object_name    
            transform (callable, optional): Optional transform 
                to be applied on a sample.
            proper_area (list): Areas to be used in order to create the proper dataset
                        Areas 1, 2, 3 and 4 for training
                        Area 5 for validation
                        Area 6 for test
            subset (list): A subset of sliding windows belonging to the test area
                         (mainly intended to help visualization of a single room)
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.all_objects_dict = all_objects_dict
        self.proper_area = proper_area
        self.subset = subset

        # Get a sorted list of all the sliding windows
        # Get only Pytorch files, in case other file types exist
        self.all_sliding_windows = sorted(
            [f for f in os.listdir(path_to_current_sliding_windows_folder) if ".pt" in f])

        self.sliding_windows = []
        
        # If we're going to visualize, the sliding windows are already known and passed
        if subset:
            self.sliding_windows = subset
        
        else:
            # Get the sliding windows for proper area and purpose
            # sliding windows for training: Area 1, Area 2, Area 3 and Area 4
            # sliding windows for val: Area 5
            # sliding windows for test: Area 6
            for area in self.proper_area:
                for f in self.all_sliding_windows:
                    if f.startswith(area):
                        self.sliding_windows.append(f)
    

    def __len__(self):
        """
        """
        return len(self.sliding_windows)

    def __getitem__(self, idx):
        """
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the proper sliding window name
        sliding_window_name = self.sliding_windows[idx]

        # Fetch the sliding window file, if necessary
        path_to_sliding_window_file = os.path.join(
                    path_to_current_sliding_windows_folder, 
                    sliding_window_name)

        # Open the sliding window file (they're saved as torch)
        sliding_window = torch.load(path_to_sliding_window_file,
                     map_location = torch.device(hparams["device"]))

        # Sample points to make all elements equal size for the dataloader to work 
        sliding_window = PointSampler(sliding_window, hparams['num_points_per_room']).sample()

        # The amount of cols to return per room point will depend on two factors:
        #  1) Are we taking the color into account? (No: 3 cols | Yes: 6 cols )
        #  2) Are we visualizing a single room? (No: 3 cols | Yes: 6 cols)
        # If we're visualing, we need to take (x_rel y_rel z_rel) and (x y z)
        # (x y z) will be saved for plotting from the original room, if they match an object
        # Sliding window cols: (x_rel y_rel z_rel R G B x y z win_ID label)
        # - 3 relative normalized points (x_rel y_rel z_rel)
        # - 3 colors (R G B)
        # - 3 absolute coordinates (x y z)
        # - 1 sliding window identifier (win_ID)
        # - 1 point label for that point (label)
        
        # Slicing the tensor 
        # [start_row_index:end_row_index, start_column_index:end_column_index]
        if self.subset:
            # TODO: Correct when tri_points_out is corrected
            # cols_to_select = [0, 1, 2, 3, 4, 5]
            sliding_window_points = sliding_window[ :, :6]
            #cols_to_select = torch.tensor([0, 1, 2, 6, 7])
            #sliding_window_points = torch.index_select(sliding_window, 1, cols_to_select)

        else:
            sliding_window_points = sliding_window[ :, :hparams["dimensions_per_object"]]

        point_labels = sliding_window[ :, -1]
        
        # Adapt the labels to num classes
        # By default, points in the sliding windows have all labels (14)
        # We must adapt the point_labels tensor to have only the proper classes:
        # movable objects: 5 + clutter
        # structural objects: 8 + clutter
        # all objects: 13 + clutter
        point_labels = AdaptNumClasses(point_labels, self.all_objects_dict).adapt()

        return sliding_window_points, point_labels
    

    def __str__(self) -> str:
        """
        """
        
        msg_list = []
        msg_list.append(80 * "-")
        msg_list.append("S3DIS DATASET INFORMATION ({})".format(self.__class__.__name__))
        msg_list.append(80 * "-")
        msg_list.append("Summary file: {}".format(eparams['s3dis_summary_file']))
        msg_list.append("Data source folder: {} ".format(self.root_dir))
        msg_list.append("Chosen areas: {} ".format(self.proper_area))
        msg_list.append("Total dataset elements: {} (from a grand total of {})".format(
            len(self.sliding_windows),
            len(self.all_sliding_windows)))
        
        if not self.subset:
            # Create a dict to know which sliding window files are per area
            sliding_windows_per_area = dict()
            for area in self.proper_area:
                    sliding_windows_per_area[area] = [f for f in self.sliding_windows if f.startswith(area)]

            for k,v in sliding_windows_per_area.items():
                msg_list.append("From {} : {} ({}...{})".format(k, len(v), v[:3], v[-3:]))     

        msg = '\n'.join(msg_list)
        msg += "\n"

        return str(msg)


class S3DISDataset4SegmentationTrain(S3DISDataset4SegmentationBase):
    """
    Augmented S3DISDataset4SegmentationBase class to create the dataset used
    for training
    """
    def __init__(self, root_dir, all_objects_dict, transform = None, proper_area = None):
        S3DISDataset4SegmentationBase.__init__(self, root_dir, all_objects_dict, transform = None, proper_area = training_areas)

class S3DISDataset4SegmentationVal(S3DISDataset4SegmentationBase):
    """
    Augmented S3DISDataset4SegmentationBase class to create the dataset used
    for validation
    """
    def __init__(self, root_dir, all_objects_dict, transform = None, proper_area = None):
        S3DISDataset4SegmentationBase.__init__(self, root_dir, all_objects_dict, transform = None, proper_area = val_areas)
    
class S3DISDataset4SegmentationTest(S3DISDataset4SegmentationBase):
    """
    Augmented S3DISDataset4SegmentationBase class to create the dataset used
    for test
    """
    def __init__(self, root_dir, all_objects_dict, transform = None, proper_area = None):
        S3DISDataset4SegmentationBase.__init__(self, root_dir, all_objects_dict, transform = None, proper_area = test_areas)

class S3DISDataset4SegmentationVisualization(S3DISDataset4SegmentationBase):
    """
    Augmented S3DISDataset4SegmentationBase class to create the dataset used
    for test
    """
    def __init__(self, root_dir, all_objects_dict, subset = None):
        S3DISDataset4SegmentationBase.__init__(self, root_dir, all_objects_dict, subset = subset)
