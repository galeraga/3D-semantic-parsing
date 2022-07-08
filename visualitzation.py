from settings import *
import model
from PIL import Image

from dataset import * #imports PointSampler
import numpy as np
import torch
import open3d as o3d
import pandas as pd #to read csv file

import pathlib
import calendar
import time


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
    ax.title.set_text(f'Input point cloud - Target: {label}')

    # plot transformation
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # preds, tnet_out = infer(model,sample) de moment no necessitem aquesta linea.
    points=tnet_out
    sc = ax.scatter(points[0,0,:], points[0,1,:], points[0,2,:], c=points[0,0,:] ,s=50, marker='o', cmap="viridis", alpha=0.7)
    ax.title.set_text(f'Output of "Input Transform" Detected: {preds}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if save == True:
        plt.savefig(f'C:/Users/marcc/OneDrive/Escritorio/Tnet-out-{label}.png',dpi=100)
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
    print('Forma sample', sample)
    print('Printing pc shape:')
    print(pc.shape)
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
        plt.savefig(f'C:/Users/marcc/OneDrive/Escritorio/Tnet-out-{label}.png',dpi=100)
    else:
        print('To save the fig change save=True')
    return fig


# The follow code will be deprectated in future versions ------------------------

def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )



"""
Function to visualize the object segmentation generated both by the model and the ground truth data. 

The function after the visualization creates a png file for each image generated.


Arguments:

    data: Tensor containing ground truth labelled points. 
          Each point is informed by its xyz location and rgb color data followed by the segmentation label identifier.
          Has the x y z r g b label or x y z label structure posible.

    segmentation_target_object_id: Integer, label that identifies the object type that has been segmented

    points_to_display: Tensor, containing the model segmented labelled points.
            The tensor has the x y z label structure.

    gt_label_col: Specifies the number of the column of the data tensor where is the segmentation label.

    model_label_col: Specifies the number of the column of the points_to_display tensor where is the segmentation label.

    b_model_without_label_col: bool, specifies if the model tensor 

    b_multiple_seg: bool, to visualize all segmentations given in the given tensors.

    draw_original_rgb_data: To render the original rgb color of the data tensor.

    b_hide_wall: bool, hides the points that corresponds to the wall
    
    b_hide_column: bool, hides the points that corresponds to the column

    b_show_inside_room: bool, to change camera point of view to the inside of the room

"""

# TODO: UPDATE NEW PARAMS EXPLANATION

                        # b_hide_wall = False, 
                        # b_hide_column = False,
def render_segmentation(dict_to_use = {},
                        str_area_and_office = "",
                        dict_model_segmented_points = {},
                        b_multiple_seg = False,                                              
                        draw_original_rgb_data = False,
                        b_show_inside_room = True,
                        b_show_room_points = True):
    
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
        print("------------------------------------------------------------")
        print("The model has not detected any points for the class " + vparams["str_object_to_visualize"])
        print("------------------------------------------------------------")
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
        # for k_object_model, v_object_model in dict_of_tensors_allpoints_per_object.items(): 
            # If founded the object with label in the dict_to_use catalog get the points
            if b_multiple_seg and k_object_model == k_object_name:
                pc_model = o3d.geometry.PointCloud()
                pc_model.points = o3d.utility.Vector3dVector(v_object_model[ :, :3]) #get xyz coordinates
                pc_model.paint_uniform_color(vparams[k_object_name + '_color'])
                all_pointcloud_object_model.append(pc_model)
    
            elif k_object_name == k_object_model and k_object_model == vparams["str_object_to_visualize"]:
                pc_model = o3d.geometry.PointCloud()
                pc_model.points = o3d.utility.Vector3dVector(v_object_model[ :, :3]) #get xyz coordinates
                pc_model.paint_uniform_color(vparams[k_object_name + '_color'])
                all_pointcloud_object_model.append(pc_model)

    vis_gt = o3d.visualization.Visualizer()
    vis_model = o3d.visualization.Visualizer()
    vis_gt.create_window(window_name='Segmentation GT id ' + vparams["str_object_to_visualize"])
    vis_model.create_window(window_name='Segmentation MODEL id ' + vparams["str_object_to_visualize"])

    # -----------------------------------
    # GT SEGMENTATION VISUALIZATION
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

        # VISUALIZATION
        # get only visualizable room points
        diameter = np.linalg.norm(np.asarray(pc_room.get_max_bound()) - np.asarray(pc_room.get_min_bound()))
        radius = diameter * 100
        camera = [0, 0, diameter]

        # Get points visible from view point
        if b_show_inside_room:
            _, pt_map = pc_room.hidden_point_removal(camera, radius)
            pc_room = pc_room.select_by_index(pt_map)

        #add pointclouds
        vis_gt.add_geometry(pc_room) 
        vis_model.add_geometry(pc_room)

    # add only the segmented objects from the GT file that are being studied
    for segment_gt in all_pointcloud_object_gt:
        vis_gt.add_geometry(segment_gt)

    # delete
    # for segment_gt_i in range(len(all_pointcloud_object_gt)):
        # if b_hide_wall and fparams["wall"] == 4:
        #     continue
        # if b_hide_column and fparams["column"] == 11:
        #     continue
        # vis_gt.add_geometry(all_pointcloud_object_gt[segment_gt_i])

    #camera point of view
    # ctr = vis_gt.get_view_control()
    # parameters = o3d.io.read_pinhole_camera_parameters("camera3.json")
    # ctr.convert_from_pinhole_camera_parameters(parameters)
    vis_gt.run()

    ts = calendar.timegm(time.gmtime())

    #save image
    vis_gt.poll_events()
    vis_gt.update_renderer()
    if b_multiple_seg:
        filename = str(ts) + '_' + str(str_area_and_office) + '__seg_GT_ALL.png'
    else:
        filename = str(ts) + '_' + str(str_area_and_office) + '__seg_GT_' +  str(vparams["str_object_to_visualize"]) + '.png'

    vis_gt.capture_screen_image(str(pathlib.Path().resolve()) + '/' + filename)

    #close window
    vis_gt.destroy_window()

    # -----------------------------------
    # MODEL SEGMENTATION VISUALIZATION
    # -----------------------------------
    # for segment_model_i in range(len(all_pointcloud_object_model)):   
    #     # if b_hide_wall and segment_model_i == 4:
    #     #     continue
    #     # if b_hide_column and segment_model_i == 11:
    #     #     continue
    #     vis_model.add_geometry(all_pointcloud_object_model[segment_model_i])

    # add only the segmented objects from the GT file that are being studied
    for segment_model in all_pointcloud_object_model:
        vis_gt.add_geometry(segment_model)

    #camera point of view
    # ctr = vis_model.get_view_control()
    # parameters = o3d.io.read_pinhole_camera_parameters("camera3.json")
    # ctr.convert_from_pinhole_camera_parameters(parameters)
    vis_model.run()

    #save image
    vis_model.poll_events()
    vis_model.update_renderer()

    if b_multiple_seg:
        filename = str(ts) + '_' + str(str_area_and_office) + '__seg_MODEL_ALL.png'
    else:
        filename = str(ts) + '_' + str(str_area_and_office) + '__seg_MODEL_' +  str(vparams["str_object_to_visualize"]) + '.png'


    vis_model.capture_screen_image(str(pathlib.Path().resolve()) + '/' + filename)

    #close window
    vis_model.destroy_window()    
