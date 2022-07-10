from settings import *
# PointSampler needed
from dataset import * 

#------------------------------------------------------------------------------
# T-NET OUTPUT VISUALIZATION
#------------------------------------------------------------------------------
def infer(model,
          points,
          shuffle_points=False,
          plot_tNet_out=True,
          return_indices_maxpool=False):
    
    '''
    This function allows to return the prediction of the class given a pointcloud.
    Parameters
    ----------
    model(model of the network):
        The model that we will pass.
    points(np array):
        The pointcloud that we want to infer saved in a .txt
    shuffle_points(bool, Default = False):
        Not implemented.
        This allows to make a permutation between the points.
    plot_tNet_out(bool, Default = True):
        Not implemented.
        Plots the tNet
    return_indices_maxpool(bool, Defalut = False):
        If True returns also the indices of the maxpool operation
    Returns
    -------
    preds(numpy array):
        An array with the predictions of our pointCloud. Each number represents the class.
    tnet_out(numpy array):
        An array with the points of our pointCloud multiplicated by the output of the T-Net.
        In other words the points displayed in a canonic way. 
    '''
    if type(points)==tuple:
        print('Type is tuple')
    
    else:
        points = points.to(hparams["device"])
    
    # We ran out of memory in GCP GPU, so all tensors have to be on the same device
    #if torch.cuda.is_available():
    #    points = points.cuda()
    #    model.cuda()

    points = points.unsqueeze(dim=0)
    model = model.eval()
    preds, feature_transform, tnet_out, ix = model(points)
    preds = preds.data.max(1)[1]

    points = points.cpu().numpy().squeeze()
    preds = preds.cpu().numpy()

    if return_indices_maxpool:
        return preds, tnet_out, ix

    return preds, tnet_out


def tnet_compare(sample, preds, tnet_out, save=False):
    '''
    Comparing this function compares a SINGLE pointCloud with the same PointCloud multiplied by the T-net.
    Parameters:
    -----------
    sample(Torch tensor):
        The sample is the object of the dataset that we want to visualize.
    preds(numpy array):
        An array with the predictions of our pointCloud. Each number represents the class.
    tnet_out(numpy array):
        An array with the points of our pointCloud multiplicated by the output of the T-Net.
        In other words the points displayed in a canonic way.
    save (Bool) Default = False:
        If True saves the image.
    Returns:
    --------
    VOID.
    '''
    # Plot 7 samples
    fig = plt.figure(figsize=[12,6]) # height and width, DO NOT CHANGE.

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # plot input sample
    #pc = sample[0].numpy()
    pc = sample.numpy()
    label = sample[1]
    sc = ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=pc[:,0] ,s=50, marker='o', cmap="viridis", alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlim3d(-1, 1)
    ax.title.set_text(f'Input point cloud')

    # plot transformation
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # preds, tnet_out = infer(model,sample) de moment no necessitem aquesta linea.
    points=tnet_out
    sc = ax.scatter(points[0,0,:], points[0,1,:], points[0,2,:], c=points[0,0,:] ,s=50, marker='o', cmap="viridis", alpha=0.7)
    ax.title.set_text(f'Point cloud in our canonical form')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if save == True:
        plt.savefig(tnet_outputs_folder + "/Tnet-out-{}.png".format(label),dpi=100)
    else:
        print('To save the fig change save=True')
    return fig


def tnet_compare_infer(model, sample, save=False):
    '''
    Comparing this function compares a SINGLE pointCloud with the same PointCloud multiplied by the T-net.
    This function is used when you don't have the tnet_out and preds.
    Parameters:
    -----------
    model(model of the network):
        The model used to infer.
    sample(Torch tensor):
        The sample is the object of the dataset that we want to visualize.
    save (Bool) Default = False:
        If True saves the image.
    Returns:
    --------
    VOID.
    '''
    # Plot 7 samples
    fig = plt.figure(figsize=[12,6]) # height and width, DO NOT CHANGE.

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # plot input sample
    #pc = sample[0].numpy()
    pc = sample.numpy()
    label = sample[1]
    sc = ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=pc[:,0] ,s=50, marker='o', cmap="viridis", alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlim3d(-1, 1)
    ax.title.set_text(f'Input point cloud - Target: {label}')

    # plot transformation
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    preds, tnet_out = infer(model,sample)
    points=tnet_out
    sc = ax.scatter(points[0,0,:], points[0,1,:], points[0,2,:], c=points[0,0,:] ,s=50, marker='o', cmap="viridis", alpha=0.7)
    ax.title.set_text(f'Output of "Input Transform" Detected: {preds}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if save == True:
        plt.savefig(tnet_outputs_folder + "/Tnet-out-{}.png".format(label),dpi=100)
    else:
        print('To save the fig change save=True')
    return fig

#------------------------------------------------------------------------------
# VISUALIZE SEGMENTATION
#------------------------------------------------------------------------------

def render_segmentation(dict_to_use = {},
                        str_area_and_office = "",
                        dict_model_segmented_points = {},
                        b_multiple_seg = False,    
                        b_hide_wall = True,                                  
                        draw_original_rgb_data = False,
                        b_show_room_points = True):
                        
    """
    Function to visualize the object segmentation generated both by the model and the ground truth data. 
    The function after the visualization creates a png file for each image generated.
    Arguments:
        dict_to_use: Dictonary with the objects to detect 
        str_area_and_office: String, area and office
        dict_model_segmented_points: Dictionary with the predicted points per slidding window per object
        b_multiple_seg: bool, to visualize all segmentations given in the dict_to_use param
        b_hide_wall: bool, hides the points that corresponds to the wall    
        draw_original_rgb_data: To render the original rgb color of the data tensor.
        b_show_room_points: Boolean, show room points, in grey color if draw_original_rgb_data is set to False
    """                        

    torch.set_printoptions(profile="full")

    
    # Open proper annotated file with the room with GT segmented points 
    a = str_area_and_office.split('_')
    str_area = a[0] + "_" + a[1]
    str_room = a[2] + "_" + a[3]

    path_to_space = os.path.join(eparams["pc_data_path"], str_area, str_room) #C:\Users\Lluis\Desktop\Projecte2\Stanford3dDataset\Area_1\conferenceRoom_1    
    sem_seg_file = str_room + eparams["pc_file_extension_sem_seg_suffix"] + eparams["pc_file_extension"] #conferenceRoom_1_annotated.txt    
    path_to_room_annotated_file = os.path.join(path_to_space, sem_seg_file) # C:\Users\Lluis\Desktop\Projecte2\Stanford3dDataset\Area_1\conferenceRoom_1\conferenceRoom_1_annotated.txt

    data = np.genfromtxt(path_to_room_annotated_file, 
                dtype = float, 
                skip_header = 1, 
                delimiter = '', 
                names = None) 

    # Stablish maximal number of points per GT file
    room_points = PointSampler(data, vparams["num_max_points_from_GT_file"]).sample()

    # create a dictionary of objects with all points from all sliding window 
    dict_of_tensors_allpoints_per_object = {}
    for slid_wind_key, slid_wind_value in dict_model_segmented_points.items():    
        for object in slid_wind_value:
            if object[-1].numel() != 0:  #check if tensor has points     

                #visualization for all objects
                if b_multiple_seg:                        
                    if object[0] in dict_of_tensors_allpoints_per_object.keys(): #check if object already in dictionary
                        dict_of_tensors_allpoints_per_object[object[0]] = \
                            torch.cat((dict_of_tensors_allpoints_per_object[object[0]], object[-1]), 0)
                    # first time to store the point tensor
                    else:
                        dict_of_tensors_allpoints_per_object[object[0]] = object[-1]                        

                # Visualization for only one object, only store study object
                elif object[0] == vparams["str_object_to_visualize"]:
                    if object[0] in dict_of_tensors_allpoints_per_object.keys(): #check if object already in dictionary
                        dict_of_tensors_allpoints_per_object[object[0]] = \
                            torch.cat((dict_of_tensors_allpoints_per_object[object[0]], object[-1]), 0)
                    # first time to store the point tensor
                    else:
                        print('passa per inici')
                        dict_of_tensors_allpoints_per_object[object[0]] = object[-1]                        

    if len(dict_of_tensors_allpoints_per_object.keys()) == 0: 
        print(80 * "-")
        print("The model has not detected any points for the class " + vparams["str_object_to_visualize"])
        print(80 * "-")
        return


    # reduce number of points per object
    dict_of_tensors_allpoints_per_object_reduced = {}
    for k_object, v_object in dict_of_tensors_allpoints_per_object.items():
        dict_of_tensors_allpoints_per_object_reduced[k_object] = \
            PointSampler(v_object, vparams["num_max_points_1_object_model"]).sample()    

    #--------------
    # GROUND TRUTH
    #--------------
    all_pointcloud_object_gt =  []
   
    for k_object_name, v_object_index in dict_to_use.items():
        if b_multiple_seg: 
            if b_hide_wall and k_object_name == "wall":
                continue
            # Get the point coordinates that matches with the object name
            points = room_points[(room_points[:, 6] == v_object_index).nonzero().squeeze(1)]
            # Create pointcloud and add it to be drawn later
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[ :, :3]) #get xyz coordinates
            pc.paint_uniform_color(vparams[k_object_name + '_color'])
            all_pointcloud_object_gt.append(pc)
            

        elif k_object_name == vparams["str_object_to_visualize"]:
            points = room_points[(room_points[:, 6] == v_object_index).nonzero().squeeze(1)]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[ :, :3]) #get xyz coordinates
            pc.paint_uniform_color(vparams[k_object_name + '_color'])
            all_pointcloud_object_gt.append(pc)

    #--------------
    # MODEL
    #--------------
    all_pointcloud_object_model =  []

    # loop through all objects in all objects stablished   
    for k_object_name, v_object_index in dict_to_use.items():
        # loop through all objects from the segmented points detected by the model
        for k_object_model, v_object_model in dict_of_tensors_allpoints_per_object_reduced.items(): 
            # If founded the object with label in the dict_to_use catalog get the points
            if b_multiple_seg and k_object_model == k_object_name:
                if b_hide_wall and k_object_model == "wall":
                    continue
                pc_model = o3d.geometry.PointCloud()
                pc_model.points = o3d.utility.Vector3dVector(v_object_model[ :, :3]) #get xyz coordinates
                pc_model.paint_uniform_color(vparams[k_object_name + '_color'])
                all_pointcloud_object_model.append(pc_model)

    
            elif k_object_name == k_object_model and k_object_model == vparams["str_object_to_visualize"]:
                pc_model = o3d.geometry.PointCloud()
                pc_model.points = o3d.utility.Vector3dVector(v_object_model[ :, :3]) #get xyz coordinates
                pc_model.paint_uniform_color(vparams[k_object_name + '_color'])
                all_pointcloud_object_model.append(pc_model)

    # timestamp as reference to stored files
    ts = calendar.timegm(time.gmtime())

    vis_gt = o3d.visualization.Visualizer()
    vis_gt.create_window(window_name='Segmentation GT id ' + vparams["str_object_to_visualize"])
    
    vis_model = o3d.visualization.Visualizer()
    vis_model.create_window(window_name='Segmentation MODEL id ' + vparams["str_object_to_visualize"])

    # -----------------------------------
    # GT AND ROOM SEGMENTATION VISUALIZATION
    # -----------------------------------

    # ROOM
    if b_show_room_points:
        # create object pointcloud
        pc_room= o3d.geometry.PointCloud()    
        pc_room.points = o3d.utility.Vector3dVector(room_points[ :, :3])

        if draw_original_rgb_data:
            # color room as original colors
            pc_room.colors = o3d.utility.Vector3dVector(room_points[ :, 3:6])
        else:
            # color to grey all room points
            pc_room.paint_uniform_color(cparams['Grey'])

        #add pointclouds
        vis_gt.add_geometry(pc_room) 
        vis_model.add_geometry(pc_room)

    # add only the segmented objects from the GT file that are being studied
    for segment_gt in all_pointcloud_object_gt:
        vis_gt.add_geometry(segment_gt)



    # ----------------------------
    # GT camera point of view 1
    # ----------------------------
    ctr = vis_gt.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters_camera1)

    #save image
    vis_gt.poll_events()
    vis_gt.update_renderer()
    vis_gt.run()

    filename = get_file_name(ts, str_area_and_office, b_multiple_seg, True, "PV1", "_hidden_wall_")
    vis_gt.capture_screen_image(camera_folder+ '/' + filename)

    # ----------------------------
    # GT camera point of view 2
    # ----------------------------
    ctr = vis_gt.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters_camera2)

    #save image
    vis_gt.poll_events()
    vis_gt.update_renderer()
    vis_gt.run()

    filename = get_file_name(ts, str_area_and_office, b_multiple_seg, True, "PV2", "_hidden_wall_")
    vis_gt.capture_screen_image(camera_folder + '/' + filename)

    #close window
    vis_gt.destroy_window()


    # add only the segmented objects from the GT file that are being studied
    for segment_model in all_pointcloud_object_model:
        vis_model.add_geometry(segment_model)

    # ---------------------
    # MODEL camera point of view 1
    # ---------------------
    ctr = vis_model.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters_camera1)

    #save image
    vis_model.poll_events()
    vis_model.update_renderer()
    vis_model.run()

    filename = get_file_name(ts, str_area_and_office, b_multiple_seg, False, "PV1", "_hidden_wall_")
    vis_model.capture_screen_image(camera_folder + '/' + filename)

    # ---------------------
    # MODEL camera point of view 2
    # ---------------------
    ctr = vis_model.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters_camera2)

    #save image
    vis_model.poll_events()
    vis_model.update_renderer()
    vis_model.run()

    filename = get_file_name(ts, str_area_and_office, b_multiple_seg, False, "PV2", "_hidden_wall_")
    vis_model.capture_screen_image(camera_folder + '/' + filename)

    #close window
    vis_model.destroy_window()    

def get_file_name(timestamp, 
        str_area_and_office = "",
        b_multiple_seg = False,
        b_is_GT_file = False,
        str_PV_version = "",
        str_sufix_hidden_wall = ""):

    if b_is_GT_file:

        if b_multiple_seg:
            return str(timestamp) + '__' + str(str_area_and_office) +  "_" + \
                        str(hparams["dimensions_per_object"]) +  "_dims_" +  \
                        str(hparams["num_classes"]) +  "_clases_" + str_sufix_hidden_wall + \
                        "_seg_GT_" + str_PV_version + ".png"
        else:
            return str(timestamp) + '__' + str(str_area_and_office) + \
                        str(hparams["dimensions_per_object"]) +  "_dims_" +  \
                        str(hparams["num_classes"]) +  "_clases_" + str_sufix_hidden_wall + \
                        "_seg_" +   str(vparams["str_object_to_visualize"]) + "_ " + \
                        "_GT_" + str_PV_version + ".png"
    else:
        if b_multiple_seg:
            return str(timestamp) + '__' + str(str_area_and_office) +  "_" + \
                        str(hparams["num_points_per_room"]) +  "_room_points_" +  \
                        str(hparams["dimensions_per_object"]) +  "_dims_" +  \
                        str(hparams["num_classes"]) +  "_clases_" + str_sufix_hidden_wall + \
                        str(hparams["epochs"]) + "_epochs_seg_" + \
                        "_MODEL_" + str_PV_version + ".png"
        else:
            return str(timestamp) + '__' + str(str_area_and_office) + \
                    str(hparams["num_points_per_room"]) +  "_room_points_" +  \
                        str(hparams["dimensions_per_object"]) +  "_dims_" +  \
                        str(hparams["num_classes"]) +  "_clases_" +  \
                        str(hparams["epochs"]) + "_epochs_" + str_sufix_hidden_wall + \
                        str(vparams["str_object_to_visualize"]) + \
                        "seg_" + \
                        "_MODEL_" + str_PV_version + ".png"  