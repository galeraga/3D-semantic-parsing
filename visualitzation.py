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


def render_segmentation(data, 
                        segmentation_target_object_id,
                        points_to_display,
                        gt_label_col = 3,
                        model_label_col = 3,
                        b_model_without_label_col = True,
                        b_multiple_seg = False,
                        draw_original_rgb_data = False,
                        b_hide_wall = False, 
                        b_hide_column = False,
                        b_show_inside_room = True):

    # Stablish a maximal number of points to visualize
    room_points = PointSampler(data, 100000).sample()
    object_points_model = PointSampler(points_to_display, 50000).sample()

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.5, 0, 0.5], [0.2, 1, 0], [0.7, 0, 0.7], [0.5, 0.2, 0], [0.2, 8, 0.4], [0.5, 1, 1], [0.5, 0.1, 0.6]]
    
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


def render_segmentation(data, 
                        segmentation_target_object_id,
                        points_to_display,
                        gt_label_col = 3,
                        model_label_col = 3,
                        b_model_without_label_col = True,
                        b_multiple_seg = False,
                        draw_original_rgb_data = False,
                        b_hide_wall = False, 
                        b_hide_column = False,
                        b_show_inside_room = True):

    # Stablish a maximal number of points to visualize
    room_points = PointSampler(data, 100000).sample()
    object_points_model = PointSampler(points_to_display, 50000).sample()

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.5, 0, 0.5], [0.2, 1, 0], [0.7, 0, 0.7], [0.5, 0.2, 0], [0.2, 8, 0.4], [0.5, 1, 1], [0.5, 0.1, 0.6]]
    
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
def render_segmentation(segmentation_target_object_id = 3,
                        str_area_and_office = "",
                        dict_model_segmented_points = {},
                        b_multiple_seg = False,
                        draw_original_rgb_data = False,
                        b_hide_wall = False, 
                        b_hide_column = False,
                        b_show_inside_room = True):
    
    # Open proper annotated file with the room with GT segmented points 
    str_area = str_area_and_office.split('_')[0]
    str_room = str_area_and_office.split('_')[1]

    path_to_space = os.path.join(eparams["pc_data_path"], str_area, str_room) #C:\Users\Lluis\Desktop\Projecte2\Stanford3dDataset\Area_1\conferenceRoom_1    
    sem_seg_file = str_room + eparams["pc_file_extension_sem_seg_suffix"] + eparams["pc_file_extension"] #conferenceRoom_1_annotated.txt    
    path_to_room_annotated_file = os.path.join(path_to_space, sem_seg_file) # C:\Users\Lluis\Desktop\Projecte2\Stanford3dDataset\Area_1\conferenceRoom_1\conferenceRoom_1_annotated.txt

    data = np.genfromtxt(path_to_room_annotated_file, 
                dtype = float, 
                skip_header = 1, 
                delimiter = '', 
                names = None) 

    # Get the data and labels arrays
    data_points_with_color = data[ :, :6]
    data_points = data[ :, :6]
    point_labels = data[ :, -1]     

    # get the tensor with all model segmenteded points
    kk_temp = []
    for key_sw, value_sw in dict_model_segmented_points.items():
        kk_temp.append(value_sw[segmentation_target_object_id][-1])
    points_to_display = torch.cat(kk_temp)

    # Stablish a maximal number of points to visualize
    #TODO define maximal number of points at settings
    room_points = PointSampler(data, 50000).sample()

    object_points_model = PointSampler(points_to_display, 50000).sample()

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.5, 0, 0.5], [0.2, 1, 0], [0.7, 0, 0.7], [0.5, 0.2, 0], [0.2, 8, 0.4], [0.5, 1, 1], [0.5, 0.1, 0.6]]
    
    #--------------
    # GROUND TRUTH
    #--------------
    all_pointcloud_object_gt =  []

    if b_multiple_seg:     
        for i in range(12+1):
            points = room_points[(room_points[:, gt_label_col] == i).nonzero().squeeze(1)]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points[ :, :3]) #get xyz coordinates
            pc.paint_uniform_color(colors[i])
            all_pointcloud_object_gt.append(pc)
    else:
        points = room_points[(room_points[:, gt_label_col] == segmentation_target_object_id).nonzero().squeeze(1)]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[ :, :3]) #get xyz coordinates
        pc.paint_uniform_color(colors[segmentation_target_object_id])
        all_pointcloud_object_gt.append(pc)

    #--------------
    # MODEL
    #--------------
    all_pointcloud_object_model =  []

    if b_multiple_seg and not b_model_without_label_col:     
        for i in range(12+1):
            pc_model = o3d.geometry.PointCloud()
            points_model = object_points_model[(object_points_model[:, model_label_col] == i).nonzero().squeeze(1)]
            pc_model.points = o3d.utility.Vector3dVector(points_model[ :, :3]) #get xyz coordinates
            pc_model.paint_uniform_color(colors[i])
            all_pointcloud_object_model.append(pc_model)
    else:
        pc_model = o3d.geometry.PointCloud()
        if b_model_without_label_col:            
            pc_model.points = o3d.utility.Vector3dVector(object_points_model[ :, :3]) #get xyz coordinates
        else:
            points_model = object_points_model[(object_points_model[:, model_label_col] == segmentation_target_object_id).nonzero().squeeze(1)]
            pc_model.points = o3d.utility.Vector3dVector(points_model[ :, :3]) #get xyz coordinates

        pc_model.paint_uniform_color(colors[segmentation_target_object_id]) #color of model and GT has to be the same  
        all_pointcloud_object_model.append(pc_model)

    #--------------
    # ROOM
    #--------------
    #create object pointcloud
    pc_room= o3d.geometry.PointCloud()    
    pc_room.points = o3d.utility.Vector3dVector(room_points[ :, :3])

    if draw_original_rgb_data:
        #color room as original colors
        pc_room.colors = o3d.utility.Vector3dVector(room_points[ :, 3:6])
    else:
        #colorize to grey all room points
        pc_room.paint_uniform_color([142/255, 142/255, 142/255])

    # -----------------------------------
    #  VISUALIZATION
    # -----------------------------------

    #-----------------------------------------
    # get only visualizable room points
    #-----------------------------------------
    diameter = np.linalg.norm(np.asarray(pc_room.get_max_bound()) - np.asarray(pc_room.get_min_bound()))
    radius = diameter * 100
    camera = [0, 0, diameter]

    # Get points visible from view point
    if b_show_inside_room:
        _, pt_map = pc_room.hidden_point_removal(camera, radius)
        pc_room = pc_room.select_by_index(pt_map)

    # -----------------------------------
    # GT SEGMENTATION VISUALIZATION
    # -----------------------------------
    vis_gt = o3d.visualization.Visualizer()
    vis_gt.create_window(window_name='Segmentation GT id ' + str(segmentation_target_object_id))
    #add pointclouds
    vis_gt.add_geometry(pc_room)
    for segment_gt_i in range(len(all_pointcloud_object_gt)):
        if b_hide_wall and segment_gt_i == 4:
            continue
        if b_hide_column and segment_gt_i == 11:
            continue
        vis_gt.add_geometry(all_pointcloud_object_gt[segment_gt_i])

    #camera point of view
    ctr = vis_gt.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("camera3.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis_gt.run()

    ts = calendar.timegm(time.gmtime())

    #save image
    vis_gt.poll_events()
    vis_gt.update_renderer()
    filename = str(ts) + '__seg_GT_label_' + str(segmentation_target_object_id) + '.png'
    vis_gt.capture_screen_image(str(pathlib.Path().resolve()) + '/' + filename)

    #close window
    vis_gt.destroy_window()

    # -----------------------------------
    # MODEL SEGMENTATION VISUALIZATION
    # -----------------------------------
    vis_model = o3d.visualization.Visualizer()
    vis_model.create_window(window_name='Segmentation MODEL id ' + str(segmentation_target_object_id))
    vis_model.add_geometry(pc_room)
    for segment_model_i in range(len(all_pointcloud_object_model)):   
        if b_hide_wall and segment_model_i == 4:
            continue
        if b_hide_column and segment_model_i == 11:
            continue
        vis_model.add_geometry(all_pointcloud_object_model[segment_model_i])

    #camera point of view
    ctr = vis_model.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters("camera3.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis_model.run()

    #save image
    vis_model.poll_events()
    vis_model.update_renderer()
    filename = str(ts) + '__seg_MODEL_label_' + str(segmentation_target_object_id) + '.png'
    vis_model.capture_screen_image(str(pathlib.Path().resolve()) + '/' + filename)

    #close window
    vis_model.destroy_window()    
