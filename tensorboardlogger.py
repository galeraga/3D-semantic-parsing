
from settings import *

class TensorBoardLogger():

    def __init__(self, args):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join(eparams['pc_data_path'], 
            eparams['tensorboard_log_dir'],
            f"{''.join(args.goal)}-{''.join(args.task)}-{''.join(args.load)}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer = SummaryWriter(logdir)
        
    def log_dataset_stats(self, summary_file):
        """
        Get insights from the dataset info, based on the ground truth
        (aka summary) file
        """
        # Open the CSV summary file
        path_to_summary_file = os.path.join(eparams['pc_data_path'], eparams['s3dis_summary_file'])

        # Get the whole summary
        summary = pd.read_csv(path_to_summary_file, 
            header =0, 
            usecols = summary_file.S3DIS_summary_cols, 
            sep = "\t"
            )
        
        #-----------------------
        # Per area info
        #-----------------------
        # Get a list of all the different areas ([Area_1, Area_2,...])
        areas = sorted(set(summary['Area']))
        # Get a list of all the unique spaces ([WC_1, WC_2,...])
        # regardless the Area where are lcoated
        unique_spaces = sorted(set(summary['Space']))
       
        total_spaces = 0
        for idx, area in enumerate(areas):  

            # Total points per area
            area_df = summary.loc[summary['Area'] == area] 
    
            self.writer.add_scalar("S3DIS Dataset/Total points per area", 
                area_df["Object Points"].sum(),
                idx + 1
                )

            for space in unique_spaces:

                # Returns a new dataframe containing all the info from
                # the Space_Z in Area_N (e.g: WC_1 in Area_1)
                space_in_area_df = summary.loc[(summary['Area'] == area) & (summary['Space'] == space)]

                # If does exist that space in that area
                if space_in_area_df.size != 0:
                    total_spaces += 1
                       
                    # Show points per room
                    self.writer.add_scalar("S3DIS Dataset/Total points per room", 
                        space_in_area_df["Object Points"].sum(),
                        total_spaces
                        )
                    # Show different classes per room
                    self.writer.add_scalar("S3DIS Dataset/Object classes per room", 
                        len(sorted(set(space_in_area_df["Object ID"]))), 
                        total_spaces
                        )             

        #-----------------------
        # Per space info
        #-----------------------
        # Get a list of all unique spaces ([hallway, WC, office...])
        all_space_labels = sorted(set(summary['Space Label']))
        already_visited_space = set()

        # Aux var to monitor the calculation
        # all_space_labels_dict = {s:(0,0) for s in all_space_labels}
        
        for idx, space in enumerate(all_space_labels):
            # Returns a new dataframe containing only the proper space
            space_labels_df = summary.loc[summary['Space Label'] == space]
            
            # all_space_labels_dict[space] = (space_labels_df["Object Points"].sum(), len(space_labels_df))

            # Finding out how many rooms there're per label type (e.g, WC)
            # Get Area_1_WC_1, Area_2_WC_1, Area_2_WC_2,...            
            for r in space_labels_df:
                area = r[0]
                space_in_area = r[1]
                already_visited_space.add(area + "_"  + space_in_area)
            
            # Mean points per space/room
            self.writer.add_scalar("S3DIS Dataset/Mean points per room type", 
                space_labels_df["Object Points"].sum()/len(already_visited_space), 
                idx
                )

        #-----------------------
        # Per object class info
        #-----------------------
        # Get a list of all the different object classes ([0, 1, ...])
        obj_classes = sorted(set(summary['Object ID'])) 

        for id in obj_classes:
            # Returns a new dataframe containing only the proper object id
            obj_class_id_df = summary.loc[summary['Object ID'] == id]
                   
            # Mean points of that object class         
            self.writer.add_scalar("S3DIS Dataset/Mean points per object class", 
                obj_class_id_df["Object Points"].sum()/len(obj_class_id_df),
                id
                )

    
    def log_hparams(self, params):
        """
        Log haprams for future reference
        """
        
        # add_scalar requires non string items
        hpars = [("hparams/" + k, torch.tensor(v)) for k, v in hparams.items() if not isinstance(v, str)]
        for p in hpars:
            self.writer.add_scalar(p[0], p[1])


    def finish(self):
        self.writer.close()
        # TODO: Send info to TensorBoard.dev

