
#from tensorboard import summary
from settings import *

#import summarizer

# Helper class
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



class S3DISDataset4Classification(torch.utils.data.Dataset):
    """
    S3DIS dataset for classification goals
    """
    
    def __init__(self, root_dir, transform = None):
        """
        Args:
            root_dir (string): Directory with all the point cluouds 
                and summary file
            transform (callable, optional): Optional transform 
                to be applied on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        # The ground truth file    
        self.summary_df = pd.read_csv(
                os.path.join(self.root_dir, eparams['s3dis_summary_file']), 
                header = None, 
                skiprows=1,
                sep = "\t"
            )

    def __len__(self):
        """
        Returns only the objects marked as "Good" in the summary file
        to avoid errors while processing data
        """
        healthy_objects = self.summary_df[self.summary_df[8] == "Good"]
        return len(healthy_objects) -1 

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the summary line (by idx) which contains the proper 
        # object point cloud info
        summary_line = self.summary_df.iloc[idx]

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

            return obj, torch.tensor(obj_label_id)
    
    def __str__(self) -> str:
        
        msg_list = []
        msg_list.append(80 * "-")
        msg_list.append("S3DIS DATASET INFORMATION")
        msg_list.append(80 * "-")
        msg_list.append("Data source: " + self.root_dir)
        msg_list.append("Summary file ({}) info: ".format(eparams['s3dis_summary_file']))
        msg_list.append(str(self.summary_df))        
        msg = '\n'.join(msg_list)
        
        return str(msg)


class S3DISDataset4Segmentation(torch.utils.data.Dataset):
    """
    S3DIS dataset for segmentation goals
    """
    
    def __init__(self, root_dir, transform = None):
        """
        Args:
            root_dir (string): Directory with all the point cluouds 
                and summary file
            transform (callable, optional): Optional transform 
                to be applied on a sample.
        
        From the ground truth file, we're going to get which spaces/rooms
        we have (unique_area_space_df)
        
              Area             Space
        0    Area_1              WC_1
        1    Area_1  conferenceRoom_1
        2    Area_1  conferenceRoom_2
        3    Area_1        copyRoom_1
        4    Area_1         hallway_1
        ..      ...               ...
        267  Area_6          office_7
        268  Area_6          office_8
        269  Area_6          office_9
        270  Area_6       openspace_1
        271  Area_6          pantry_1
        """
        
        self.root_dir = root_dir
        self.transform = transform
        
        # The ground truth file    
        self.summary_df = pd.read_csv(
                os.path.join(self.root_dir, eparams['s3dis_summary_file']), 
                header =0, 
                sep = "\t") 

        # Get unique area-space combinations from summary_df
        # in order to know the exact number of spaces/rooms (272)
        self.unique_area_space_df = self.summary_df[["Area", "Space"]].drop_duplicates(ignore_index=True)         

    def __len__(self):
        """
        """
        return len(self.unique_area_space_df) -1 

    def __getitem__(self, idx):
        """
        Returns all the points in that space/room/scene.

        In other words, it returns an adapted version (sampled or padded)
        of the content of the file Area_N\space_X\space_x_annotated.txt
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the summary line (by idx) which contains the proper 
        # object point cloud info
        summary_line = self.unique_area_space_df.iloc[idx]

        # Get the point cloud info from the summary line
        area = summary_line[0]
        space = summary_line[1]
          
        # Fetch the object point cloud file for the whole room
        sem_seg_file = space + eparams["pc_file_extension_sem_seg_suffix"] 
        sem_seg_file += eparams["pc_file_extension"]
        path_to_obj = os.path.join(self.root_dir, area, space, sem_seg_file )
         
        space_df = pd.read_csv(
            path_to_obj, 
            sep = "\t", 
            dtype = np.float32, 
            header = None,
            )
        # Convert the whole room file to tensor
        room = torch.tensor(space_df.values).to(hparams["device"])

        # Make all the point clouds equal size
        room = PointSampler(room, hparams['max_points_per_space']).sample()

        # The amount of cols to return per room will depend on whether or not
        # we're taking the color into account
        # room -> [x y x r g b label] (7 cols)
        room_points = room[ :, :hparams["dimensions_per_object"]]
        point_labels = room[ :, -1]
        
        return room_points, point_labels
    

    def __str__(self) -> str:
        """
        """
        
        msg_list = []
        msg_list.append(80 * "-")
        msg_list.append("S3DIS DATASET INFORMATION")
        msg_list.append(80 * "-")
        msg_list.append("Data source: " + self.root_dir)
        msg_list.append("Summary file ({}) info: ".format(eparams['s3dis_summary_file']))
        msg_list.append(str(self.summary_df))        
        msg_list.append("\nAreas and spaces info (for semantic purposes): ")
        msg_list.append(str(self.unique_area_space_df))  
        msg = '\n'.join(msg_list)
        
        return str(msg)

# TODO: The new dataset for semantic segmenation. It will replace the above one,
# once the Clara's sliding windows mehtod will be finished 
class S3DISDataset4Segmentation_(torch.utils.data.Dataset):
    """
    S3DIS dataset for segmentation goals.
    
    It works with sliding blocks

    From the XXX method, there's a folder containing all the pre-processed 
    sliding block for all the spaces/rooms in the dataset. 
    
    The folder is located in XXX and has the following name and contents:

    \root_dir\sliding_windows_overlap_ZZZ\Area_N_Space_X_Block_Y

    where:
    - ZZZ indicates the overlap percentage between consecutive 
        windows (050 -> 50%, 075 ->75%, 100 ->100% )
    
    - Block_Y: XXX

    """
    
    def __init__(self, root_dir, transform = None):
        """
        Args:
            root_dir (string): Directory with all the pre-processed 
                sliding windows (the output from XXX method)
            transform (callable, optional): Optional transform 
                to be applied on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        
        # Get a list of all the sliding windows
        
        self.all_sliding_windows = ...

    def __len__(self):
        """
        """
        return len(self.all_sliding_windows) 

    def __getitem__(self, idx):
        """
        In order to follow the original paper implmenetation, the size of this
        method output will be based on the task type:

        - If task == train, only 4096 points will be returned per sliding windows
        - If task == test, all the points in the proper sliding window will be returned

        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the proper sliding window name
        sliding_window_name = self.all_sliding_windows[idx]

        # Fetch the sliding window file, if necessary
        path_to_sliding_window_file = os.path.join(self.root_dir, sliding_window_name)

        # Open the sliding window file
        sliding_window = None

        # Convert the sliding window to torch, if necessary
        sliding_window = torch.tensor(sliding_window).to(hparams["device"])

        # For training:
        #   - We set the max points per sliding windows as per original paper (4096)
        # For testing:
        #   - If we're in a batch mode, we limit the amount of points per room to
        #     make all the rooms equal size
        max_points_per_sliding_window = hparams['max_points_per_sliding_window'] if "train" in args.task else hparams['max_points_per_space']
        
        sliding_window = PointSampler(sliding_window, max_points_per_sliding_window)

        # TODO: sliding_window -> [x y x r g b label] (7 cols) (HOW MANY COLUMNS IT WILL HAVE)
        # The amount of cols to return per room will depend on whether or not
        # we're taking the color into account
        sliding_window_points = sliding_window[ :, :hparams["dimensions_per_object"]]
        point_labels = sliding_window[ :, -1]
        
        return sliding_window_points, point_labels
    

    def __str__(self) -> str:
        """
        """

        msg_list = []
        msg_list.append(80 * "-")
        msg_list.append("S3DIS DATASET INFORMATION")
        msg_list.append(80 * "-")
        msg = '\n'.join(msg_list)
        
        return str(msg)

