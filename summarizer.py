from settings import *

class S3DIS_Summarizer():
    """
    Class to generate the ground truth file.
    It also gets additional info from the S3DIS dataset, such as 
    the points per object or the data health status
    """

    # Names of the cols are going to be saved in the CSV summary file
    # after folder traversal
    S3DIS_summary_cols = [
            "Area", 
            "Space", 
            "Space Label", 
            "Space ID", 
            "Object", 
            "Object Points", 
            "Object Label", 
            "Object ID", 
            "Health Status"
        ]
    
    def __init__(self, path_to_data, logger, rebuild = False, check_consistency = False):
        """
        Inspect the dataset to get the following info:

            - Areas
            - Spaces
            - Objects
            - Points per object
            - Labels (for both areas and spaces)
            - Data health status
        
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
            - Space label
            - Space ID 
            - Object
            - Points per object
            - Object label
            - Object ID
            - Health Status

        Args:
            - rebuild: If rebuild is set to True, the summary file is 
                generated again
            - check_consistency: perform a data consistency check in 
                order to remove data with inconsistent format
        """
        
        self.path_to_data = path_to_data
        self.rebuild = rebuild
        self.path_to_summary_file = os.path.join(self.path_to_data, eparams['s3dis_summary_file'])
        self.logger = logger

        # Do NOT process the info if the summary file already exists
        if os.path.exists(self.path_to_summary_file):
            # Creates the Pandas DataFrame for future use in the class
            self.summary_df = pd.read_csv(self.path_to_summary_file, 
                header =0, 
                usecols = self.S3DIS_summary_cols, 
                sep = "\t")     
            
            if self.rebuild == False:
                msg = "Skipping summary file generation. The S3DIS summary file {} already exists in {}"
                print(msg.format(eparams['s3dis_summary_file'], self.path_to_data))
                
            if check_consistency:
                  self.check_data_consistency()
            
            return
        
        print("Generating ground truth file (summary file) from folder traversal {} in {}".format(eparams['s3dis_summary_file'], self.path_to_data))

        # Every line of the S3DIS summary file will contain:
        # (area, space, space_label, space_id, object, 
        # points_per_object, object label, object_id, health_status)
        summary_line = []
        
        # Aux vars to assign a dict-like ID to spaces and objects
        spaces_dict = dict()
        total_spaces = 0
        objects_dict = dict()
        total_objects = 0

        # Keep only folders starting with Area_XXX
        areas = dict((folder, '') for folder in sorted(os.listdir(self.path_to_data)) if folder.startswith('Area'))

        # For every area folder, get the disjoint spaces included within it
        for area in areas:

            # os.path.join takes into account the concrete OS separator ("/", "\")        
            path_to_spaces = os.path.join(self.path_to_data, area)
            
            # Get spaces for each area, avoiding non disered files (".DStore", ...)
            spaces = sorted([space for space in os.listdir(path_to_spaces) 
                if not '.' in space])    
            
            # For every space, get the objects it contains
            for space in spaces:
                path_to_objects = os.path.join(path_to_spaces, space, "Annotations")
                
                # Get the space label
                # From hallway_1, hallway_2, take only "hallway"
                space_label = space.split("_")[0]
                
                # Update the spaces dict
                # {'hall': 0, 'wall': 1, ...}
                if space_label not in spaces_dict:
                    spaces_dict[space_label] = total_spaces
                    total_spaces += 1
                
                space_idx = spaces_dict[space_label]
                
                # The file to be used will be the original of the S3DIS 
                # (not the_rgb_norm.txt), since rgb normalization is 
                # optional (only required to visualize data with Open3D)        
                objects = sorted([object for object in os.listdir(path_to_objects) 
                    if (eparams['pc_file_extension'] in object) and (eparams['already_rgb_normalized_suffix'] not in object)])    

                desc = "Getting points from objects in {} {}".format(area, space)
                
                for object in tqdm(objects, desc = desc):
                    # Get the object label
                    # From chair_1, chair_2, take only "chair"
                    object_label = object.split("_")[0]
                    
                    # Update the object dict
                    # {'chair': 0, 'table': 1, ...}
                    if object_label not in objects_dict:
                        objects_dict[object_label] = total_objects
                        total_objects += 1
                    
                    object_idx = objects_dict[object_label]

                    # Get the number of points in the object
                    with open(os.path.join(path_to_objects, object)) as f:
                        points_per_object = len(list(f))
                    
                    # Save all the traversal info in the summary file:
                    # (Area, space, space_label, space ID, object, 
                    # points per object, object label, object ID, health status)
                    summary_line.append((area, space, space_label, space_idx, object, 
                        points_per_object, object_label, object_idx,  "Unknown"))

        # Save the data into the CSV summary file
        self.summary_df = pd.DataFrame(summary_line)
        self.summary_df.columns = self.S3DIS_summary_cols
        self.summary_df.to_csv(os.path.join(eparams['pc_data_path'], eparams['s3dis_summary_file']), index = False, sep = "\t")
        
        # Always check consistency after generating the file
        self.check_data_consistency()
        
    
    def check_data_consistency(self):
        """
        Check the point cloud files in order to know whether the files are 
        readable or not. 

        To do so, reading the CSV with pandas as np.float32 is enough to know
        whether future conversion to torch tensor will work.

        If the conversion is feasible, flag the point cloud file as 
        "Good" in the "Health Status" col of the sumamry file

        If not, flag de point cloud file as "Bad" in the "Health Status" 
        col of the sumamry file
        """

    
        for idx, row in tqdm(self.summary_df.iterrows(), 
            desc = "Checking data consistency. Please wait..."):
            
            # Get the values needed to open the physical TXT file
            summary_line = self.summary_df.iloc[idx]
            area = summary_line[0]
            space = summary_line[1]
            obj_file = summary_line[4]
        
            # print("Checking consistency of file {}_{}_{} (idx: {})".format(
            #    area, space, obj_file, idx), end =' ')

            # Fetch the object point cloud
            path_to_obj = os.path.join(self.path_to_data, area, space, "Annotations", obj_file)

            try:
                # Let's try to open the file as np.float32      
                pd.read_csv(path_to_obj, sep = " ", dtype = np.float32)
                
                # Flag the file as "Good", if success
                self.summary_df.at[idx,"Health Status"] = "Good"
                
            except ValueError:
                # Flag the file as "Bad", if failure
                self.summary_df.at[idx,"Health Status"] = "Bad"

                # Write error on logger
                msg = "The following file seems to be corrupted: {} ".format(path_to_obj)
                self.logger.writer.add_text("Summarizer/Error", msg)
                
            
            finally:
                # Save health status changes
                self.summary_df.to_csv(os.path.join(eparams['pc_data_path'], eparams['s3dis_summary_file']), index = False, sep = "\t")

    def label_points_for_semantic_segmentation(self):
        """
        Create a single annotated file (per room/space) for semantic segmentation

        Method outlook:

         - The label to be assigned to each point in the cloud will be based
            on the file name the point is (e.g. all the points in chair_1.txt
            will have the "chair" label). In fact, since the already created 
            summary file already constains this info, it will be used to get
            the proper label for each point
        
        - A new file will be created for every space/room containing all the 
            annotated points. This file will be called space_annotated.txt 
            (e.g, conferenceRoom_1_annotated.txt) and will be saved next to
            the original non-annotated space/room (e.g, Area_1\office_1\office_1_annotated.txt)
        
        - This new file:
                - will be the concatation of all the files inside the 
                "Annotations" folder (since the file Area_1\office_1\office_1.txt
                is the sum of all the files within Area_1\office_1\Annotations)
                - will have an additional column for the label (based on the
                file name we're concatenating)
        
        Following the Area_1\office_1 example, this way a new file called
        Area_1\office_1\office_1_annotated.txt will be created. 
        This file will contain an extra column to host the label for every 
        point in the cloud.

        unique_area_space_df (with 272 rows! Note the index is not correlative)

        Area_space: 
            Area             Space
        0     Area_1              WC_1
        41    Area_1  conferenceRoom_1
        73    Area_1  conferenceRoom_2
        119   Area_1        copyRoom_1
        141   Area_1         hallway_1
        ...      ...               ...
        9576  Area_6          office_7
        9614  Area_6          office_8
        9689  Area_6          office_9
        9763  Area_6       openspace_1
        9811  Area_6          pantry_1
        """

        # Get unique area-space combinations from summary_df
        # in order to know the exact number of spaces (around 272)
        unique_area_space_df = self.summary_df[["Area", "Space"]].drop_duplicates()     
        
        # Aux vars to keep track of the labeling progress
        total_unique_spaces = len(unique_area_space_df)
        processed_spaces = 0

        print("Checking whether point labeling has to be performed...")
        
        for i, (idx, row) in enumerate(unique_area_space_df.iterrows()):         
            # Get the proper area and space
            area = row["Area"]
            space = row["Space"]
            
            # Defining path for the folder where the semantic segmentaion
            # file is going to be saved
            # (e.g. Area_1\office_1\office_1_annotated.txt)
            path_to_space = os.path.join(self.path_to_data, 
                area, 
                space
                )
            
            path_to_objs = os.path.join(path_to_space,
                "Annotations"
                )
                
            sem_seg_file = space + eparams["pc_file_extension_sem_seg_suffix"] + eparams["pc_file_extension"]
            path_to_sem_seg_file = os.path.join(path_to_space, sem_seg_file)
            
            # Checking if the semantic segmentation file already exists
            # for this space within this area
            if os.path.exists(path_to_sem_seg_file):
                processed_spaces += 1
                msg = "({}/{}) Skipping generation ".format(processed_spaces, total_unique_spaces)
                msg += "of the semantic segmentation file"
                msg += " {}_{} (file {} already exists)".format(
                        area, 
                        space,
                        sem_seg_file)
                print(msg)
            
            # Create the semantic segmentation file   
            else:
                # Update the processed rooms counter
                processed_spaces += 1

                # Let's start creating an empty semantic segmentation dataframe
                sem_seg_df = pd.DataFrame()
                       
                # Get all the objects that belong to that concrete area and space
                # (NOTE: from the summary_df!)
                objects_df = self.summary_df[(self.summary_df["Area"] == area) &
                    (self.summary_df["Space"] == space)]

                # Define the message to print with tqdm
                tqdm_msg = "({}/{}) Generating file ".format(processed_spaces, total_unique_spaces) 
                tqdm_msg += "for semantic segmentation "
                tqdm_msg += "in {}_{}".format(area, space)

                # Let's read every object/class file
                # TODO: Find out a faster way to compute the loop 
                # (e.g, cudf lib, move to numpy to gpu, etc.)
                # TODO: Quicker way: 
                #   - get tha path to Area_N\Room_N\Annotations\
                #   - Reads all files in a list comprehension and make torch.stack
                #     torch.stack([torch.FloatTensor(i['img']) for i in batch])  
                #   - Save the file
                for i in tqdm(objects_df.index, desc = tqdm_msg):
                    # Get line by line info
                    summary_line = self.summary_df.iloc[i]

                    # Get the proper info from each line
                    obj_file = summary_line["Object"]
                    obj_label_id = summary_line["Object ID"]
                    
                    try:
                        # Let's try to open the file as np.float32      
                        path_to_obj = os.path.join(path_to_objs, obj_file)
                        obj_df = pd.read_csv(path_to_obj, 
                            sep = " ", 
                            header = None,
                            dtype = np.float32)
                        
                        # Adding the new col with the proper label
                        # https://stackoverflow.com/questions/42473098/add-column-to-pandas-without-headers
                        obj_df[len(obj_df.columns)] = obj_label_id

                        # Save the semantic segmentation file
                        sem_seg_df = pd.concat([sem_seg_df, obj_df])
                        sem_seg_df.to_csv(path_to_sem_seg_file, index = False, sep = "\t")
                                                                     
                    except:    
                        # Write error on logger
                        msg = "The following file seems to be corrupted: {} ".format(path_to_obj)
                        self.logger.writer.add_text("Summarizer/Error", msg)
                 

    def report_health_issues(self):
        """
        Provide information about how many objects have issues with 
        their point cloud

        
        Return the indexes of the rows/files flagged with bad healthy status
        """     
        
        return list(self.summary_df.index[self.summary_df["Health Status"] == "Bad"])

    def get_labels(self):
        """
        Get the labels from the S3DIS dataset folder structure

        Create dicts with the different spaces (conf rooms, hall ways,...)
        and objects (table, chairs,...) within an Area 
        
        Output:
            space_labels: A dict containing {space_0: 0, space_1: 1 ... }
            object_labels: A dict containing {object_0: 0, object_1:1, ... }
        """
        
        if not os.path.exists(self.path_to_summary_file):
            msg = "No S3DIS summary file {} found at {}."
            msg += "Summary file is going to be automatically generated"
            print(msg.format(eparams['s3dis_summary_file'], self.path_to_data))     
            self.__init__(self.path_to_data, rebuild = True)
        
        # Define the sets and dicts to be used 
        spaces_dict = dict()
        objects_dict = dict()

        # Get Object IDs and labels to build a dict
        # {'celing': 0, 'clutter':1,...}
        unique_objects_df = self.summary_df[["Object ID", "Object Label"]].drop_duplicates(ignore_index=True) 
        for row in unique_objects_df.iterrows():
            objects_dict[row[1][1]] = row[1][0]
        
        # Get Space IDs and labels to build a dict
        # {'WC': 0, 'conferenceRoom':1,...}
        unique_spaces_df = self.summary_df[["Space ID", "Space Label"]].drop_duplicates(ignore_index=True) 
        for row in unique_spaces_df.iterrows():
            spaces_dict[row[1][1]] = row[1][0] 

        return spaces_dict, objects_dict

        
    def get_stats(self):
        """
        Get several statistics about the dataset
        """

        if not os.path.exists(self.path_to_summary_file):
            msg = "No S3DIS summary file {} found at {}."
            msg += "Summary file is going to be automatically generated"
            print(msg.format(eparams['s3dis_summary_file'], self.path_to_data))     
            self.__init__(self.path_to_data, rebuild = True)
        
        # Open the CSV summary file
        summary = os.path.join(self.path_to_data, eparams['s3dis_summary_file'])

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
        
        # The cumulative of the quantiles.
        print("Quantile 90%: ", object_points_df.quantile(0.90))
        # quantiles and percentiles of points
        print("Quantile 10%:", object_points_df.quantile(0.10))    
        
        # Total objects per space
        #space_list = summary['Space']
        space_list = summary['Space'].to_list()
        space_set = set(space_list)
        for space in space_set:
            space_df = summary.loc[summary['Space'] == space]
            objects_per_area = len(sorted(set(space_df["Object Points"])))
            print("Space: {}, number of objects: {}".format(space, objects_per_area))
        
        # Points per area
        for area in areas:
            area_df = summary.loc[summary['Area'] == area]
            area_points = area_df["Object Points"]
            print("Points per {}: {}".format(area, area_points.sum()))

        # Points per space:
        for space in space_set:
            space_df = summary.loc[summary['Space'] == space]
            space_points = space_df["Object Points"]
            #print("Points per {}: {}".format(space, space_points.sum()))
        
        # Points per kind of space.
        # He comprovat que els valors que aquests valors son les sumes dels anteriors.
        summary_spaces = summary.groupby(summary["Space"].str.split('_').str[0]).sum()
        print(summary_spaces)

        # TODO: Points per kind of object
        # We sum all the Points of summary['Object'] that have similar names.
        summary_objects = summary.groupby(summary['Object'].str.split('_').str[0]).sum()
        print(summary_objects)

    
    def create_sliding_windows(self, rebuild = False): 
        """
        Creates the CSV files that will store all the sliding windows used in 
        the semantic segmentation training.

        Sliding window parameters are user-defined and read from settings.py.
        These params are: 
        
        w: width of the sliding window
        d: depth of the sliding window
        h: height of the sliding window
        o: overlapping of consecutives sliding window
        
        All sliding windows are created by splitting the proper room annotated
        file (\Area_N\Space_X\space_x_annotated.txt) according to the 
        user-defined params.

        In order to simplify dataset management, all the sliding windows for all
        the available rooms are saved into a single folder:
        
        Area_N
        sliding_windows
        ├── w_X_d_Y_h_Z_o_T
        ├── w_P_d_Q_h_R_o_S
        ├── ...  
        """

        # Create the folder to store the sliding windows with chosen params
        # The folder will follow this convention: w_X_d_Y_h_Z_o_T
        # path_to_root_sliding_windows_folder is set in settings.py
        chosen_params = 'w' + str(hparams['win_width']) 
        chosen_params += '_d' + str(hparams['win_depth'])
        chosen_params += '_h' + str(hparams['win_height']) 
        chosen_params + '_o' + str(hparams['overlap']) 
        
        path_to_current_sliding_windows_folder = os.path.join(
                        path_to_root_sliding_windows_folder, chosen_params)
                
        if not os.path.exists(path_to_current_sliding_windows_folder):
            os.makedirs(path_to_current_sliding_windows_folder)

        # Remove all existing sliding windows for the chosen params, if required
        if rebuild == True: 
            print("Removing contents from folder ", path_to_current_sliding_windows_folder)
            for f in tqdm(os.listdir(path_to_current_sliding_windows_folder)):
                os.remove(os.path.join(path_to_current_sliding_windows_folder, f))


        # Get unique area-space combinations from summary_df
        # in order to know the exact number of spaces (around 272) 
        unique_area_space_df = self.summary_df[["Area", "Space"]].drop_duplicates()     
        
        # Create sliding windows for every unique area-space combination
        print("Creating sliding windows for:")
        progress_bar = tqdm(unique_area_space_df.iterrows(), total = len(unique_area_space_df))
        
        for (idx, row) in progress_bar:
            # Get the proper area and space
            area = row["Area"]
            space = row["Space"]
            
            # Update the info provided in the progess bar
            progress_bar.set_description(area + "_" + space)

            # Create the sliding windows
            self.create_sliding_windows_for_a_single_room(area, space, path_to_current_sliding_windows_folder)
        
    
    def create_sliding_windows_for_a_single_room(self, area, space, folder):
        """
        """

        # For comodity's sake, put the sliding windows params in local vars
        win_width = hparams['win_width']
        win_depth = hparams['win_depth']
        win_height = hparams['win_height']
        overlap = hparams['overlap']
        overlap_fc = 100 - overlap
    
        # Open the proper annotated file
        # (e.g. Area_1\office_1\office_1_annotated.txt)
        path_to_space = os.path.join(self.path_to_data, 
            area, 
            space
            )
        
        sem_seg_file = space + eparams["pc_file_extension_sem_seg_suffix"] + eparams["pc_file_extension"]     
        
        path_to_room_annotated_file = os.path.join(path_to_space, sem_seg_file)
        
        data = np.genfromtxt(path_to_room_annotated_file, 
                    dtype = float, 
                    skip_header = 1, 
                    delimiter = '', 
                    names = None) 
    
        # TODO: Check conversion to float and torch tensor
        # data = torch.from_numpy(data).float()
        # data_arr=data.to_numpy()
        # data_arr = data
        
        # Get the data and labels tensors
        data_points = data[ :, :hparams["dimensions_per_object"]]
        point_labels = data[ :, -1] 

        # Create column vectors for each X, Y, Z coordinates
        abs_x = data_points[ :, 0]
        abs_y = data_points[ :, 1]
        abs_z = data_points[ :, 2]
        
        '''
        # FOR DEBUGGING PLOT X_Y ROOM SCATTER
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.scatter(tri_points[:,0].tolist(), tri_points[:,1].tolist())
        '''    
        
        # Find roommax_x, roommax_y, roommin_x, roommin_y, roommin_z from all points in room. 
        # Origin will be (roomin_x, roommin_y, roommin_z)
        roommax_x = max(abs_x)
        roommax_y = max(abs_y)
        roommin_x = min(abs_x)
        roommin_y = min(abs_y)
        roommin_z = min(abs_z)
                
        # Variables of window are the 4 corners of the windows in X-Y. 
        # These are defined by the combinations of the x and y max and 
        # min values of each window where: 
        #   - (winmin_x winmin_y) --> origin of window
        #   - winmax_x = winmin_x + win_width --> max values of x
        #   - winmax_y = winmin_y + win_depth --> max values of y
        
        # Slide window on x until winmax_x>roommax_x and on y until winmax_y>roommax_y

        # Define vectors of origins
        # winmax_z is defined but not used since we don't care about the height
        # (we take all points in Z)
        winmin_xvec = np.arange(roommin_x, roommax_x, overlap_fc/100*win_width)
        winmin_yvec = np.arange(roommin_y, roommax_y, overlap_fc/100*win_depth)
        winmin_z = roommin_z
        winmax_z = roommin_z + win_height 

        # normalized point matrix inside window, with origin on the window's origin 
        # (not the absolute origin), and column for window count
        # points_win=np.zeros([1,tri_points.shape[1]+1]) 
        # label matrix for points inside window, same order as points_win
        # labels_win=[] 

        # ID the number of the window with more than 0 points in the room, 
        # to separate one window from the other
        win_count = 0

        # For each possible origin of each window in the room, find the points 
        # "trapped" inside it and transform them to relative normalized coordinate
        for (winmin_x,winmin_y) in itertools.product(winmin_xvec, winmin_yvec):
            
            # Define the maximum values of x and y in that window   
            winmax_x = winmin_x + win_width
            winmax_y = winmin_y + win_depth
            
            # Get the entire room point cloud from where we will select the window points 
            tri_points_aux = data_points
            labels_aux = point_labels
            
            # Select only points that are inside the defined x limits for the specific window
            # point_sel is a True/False Matrix
            point_sel = np.array((tri_points_aux[:,0] > winmin_x) & (tri_points_aux[:,0] < winmax_x)) 
            tri_points_aux = tri_points_aux[point_sel,:]
            labels_aux = labels_aux[point_sel]

            # Select only points that are inside the defined y limits for the specific window
            point_sel = np.array((tri_points_aux[:,1] > winmin_y) & (tri_points_aux[:,1] < winmax_y))
            tri_points_aux = np.array(tri_points_aux[point_sel])
            labels_aux = labels_aux[point_sel]
            
            # If there are no points in the defined window, ignore the window
            if tri_points_aux.size != 0: 
                
                # tri_point_aux is now the matrix containing only the 3D points 
                # inside the prism window in absolute coordenates
                # Take each vector separately
                abs_x_win = tri_points_aux[:, 0]
                abs_y_win = tri_points_aux[:, 1]
                abs_z_win = tri_points_aux[:, 2]
                
                # Transform coordinates to relative (with respect to window origin, 
                # not absolute origin) and normalize with win_width, win_depth and win_height
                # rel_x, rel_y, rel_z are vectors
                rel_x = (abs_x_win-winmin_x)/win_width 
                rel_y = (abs_y_win-winmin_y)/win_depth 
                rel_z = (abs_z_win-winmin_z)/win_height

                tri_points_rel = np.copy(tri_points_aux)
                
                # Put the relative and normalized points inside a matrix with the color information
                # tri_points aux is a matrix with relative as well as rgb info
                tri_points_rel[:,0] = rel_x 
                tri_points_rel[:,1] = rel_y
                tri_points_rel[:,2] = rel_z

                # Convert to 1D array else it won't work
                labels_aux.shape=(len(labels_aux), 1) 
                
                # Create matrix with: 
                # - 3 relative normalized points, then 
                # - 3 colors, then 
                # - 3 absolute coordinates, then
                # - 1 window identifier, then 
                # - 1 label
                tri_points_out = np.concatenate((tri_points_rel, tri_points_aux[:,0:3], np.full((len(rel_x),1), win_count), labels_aux), axis = 1)

                # Convert the NumPy matrix to a float torch tensor
                tri_points_out = torch.from_numpy(tri_points_out).float()
                
                # Save the torch tensor as a file
                # Common PyTorch convention is to save tensors using .pt 
                # file extension
                sliding_window_name = area + '_' + space + "_" 
                sliding_window_name += "win" + str(win_count) + ".pt"
                torch.save(tri_points_out, os.path.join(folder, sliding_window_name))

                # Update the number of sliding windows
                win_count += 1
        
                


   

    