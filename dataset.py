
from settings import *

class S3DISDataset(torch.utils.data.Dataset):
    """
    S3DIS dataset
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the point cluouds and summary file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.summary_df = pd.read_csv(os.path.join(self.root_dir, S3DIS_SUMMARY_FILE), header = None, skiprows=1)
        self.transform = transform

    def __len__(self):
        # Minus to match the dataset split
        return len(self.summary_df) - 1

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the summary line (by idx) which contains the object point cloud 
        summary_line = self.summary_df.iloc[idx, 0]
        
        # From the sumamry line, extract the proper info
        area, space, obj_file, total_obj_points, space_label, obj_label = summary_line.split("\t")        
        
        # Fetch the object point cloud
        path_to_obj = os.path.join(self.root_dir, area, space, "Annotations", obj_file)
        obj_df = pd.read_csv(path_to_obj, sep = " ", dtype = np.float32)
        obj = torch.tensor(obj_df.values)

        # TODO: Review why we points in the cloud does not excatly match
        # print("Object shape {} (poins cloud from summary: {}) ({}_{}_Annotations_{}): ".format(obj.shape, total_obj_points, area, space, obj_file))
      
        # Torch Dataloaders expects each tensor to be equal size
        # TODO: MAX_OBJ_POINTS has te be defined, based on point cloud analysis
        if(len(obj) > MAX_OBJ_POINTS):   
            # Sample points 
            # Shuffle the index (torch.randperm(4) -> tensor([2, 1, 0, 3]))
            idxs = torch.randperm(len(obj))[:MAX_OBJ_POINTS]
            obj = obj[idxs]
        
        else:
            # Duplicate points
            obj_len = len(obj)
            padding_points = MAX_OBJ_POINTS - obj_len     
            for _ in range(padding_points):     
                duplicated_obj_points = torch.unsqueeze(obj[random.randint(0, obj_len)], dim = 0)
                obj = torch.cat((obj, duplicated_obj_points), dim = 0)
        
        if self.transform:
            obj = self.transform(obj)

        return obj, obj_label