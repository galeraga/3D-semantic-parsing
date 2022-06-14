
from tensorboard import summary
from settings import *

import summarizer

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

            obj = torch.tensor(obj_df.values)

            # Torch Dataloaders expects each tensor to be equal size
            # TODO: MAX_OBJ_POINTS has te be defined, based on point cloud analysis
            if(len(obj) > hparams['num_points_per_object']):   
                # Sample points 
                # Shuffle the index (torch.randperm(4) -> tensor([2, 1, 0, 3]))
                idxs = torch.randperm(len(obj))[:hparams['num_points_per_object']]
                obj = obj[idxs]
            
            else:
                # Duplicate points
                obj_len = len(obj)
                padding_points = hparams['num_points_per_object'] - obj_len     
                for _ in range(padding_points):     
                    duplicated_obj_points = torch.unsqueeze(obj[random.randint(0, obj_len - 1)], dim = 0)
                    obj = torch.cat((obj, duplicated_obj_points), dim = 0)
            
            if self.transform:
                obj = self.transform(obj)

            # Get the labels dict
            # {0: 'openspace', 1: 'pantry', ... , 10: 'lounge'}
            # {0: 'bookcase', 1: 'door', 2: 'ceiling', ... , 13: 'floor'}
            
            #object_labels_dict = summary.get_labels()[1]
            # Get the key from the value dict
            #obj_label = [k for k,v in object_labels_dict if v == obj_label]
            # Convert the label key to tensor
            #obj_label = torch.tensor(obj_label)
            
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
          
        # Fetch the object point cloud for the whole room
        sem_seg_file = space + eparams["pc_file_extension_sem_seg_suffix"] 
        sem_seg_file += eparams["pc_file_extension"]
        path_to_obj = os.path.join(self.root_dir, area, space, sem_seg_file )
        
        # Element order is ignored when data is retrieved by cols in Pandas
        # so we need to define the order of the cols
        cols_to_get = [col for col in range (hparams['dimensions_per_object'])]
        space_df = pd.read_csv(
            path_to_obj, 
            sep = " ", 
            dtype = np.float32, 
            header = None,
            usecols = cols_to_get 
            )[cols_to_get]

        room = torch.tensor(space_df.values)

        # Torch Dataloaders expects each tensor to be equal size
        if(len(room) > hparams['max_points_per_space']):   
            # Sample points 
            # Shuffle the index (torch.randperm(4) -> tensor([2, 1, 0, 3]))
            idxs = torch.randperm(len(room))[:hparams['max_points_per_space']]
            room = room[idxs]
        
        else:
            # Duplicate points
            room_len = len(room)
            padding_points = hparams['max_points_per_space'] - room_len     
            for _ in range(padding_points):     
                duplicated_room_points = torch.unsqueeze(room[random.randint(0, room_len - 1)], dim = 0)
                room = torch.cat((room, duplicated_room_points), dim = 0)
        
        if self.transform:
            room = self.transform(room)

        # TODO: GET THE SPACE LABEL ID FOR EACH POINT

        return room, torch.tensor(space_label_id)
    

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
        msg_list.append("Areas and spaces info (for semantic purposes): ")
        msg_list.append(str(self.unique_area_space_df))  
        msg = '\n'.join(msg_list)
        
        return str(msg)
