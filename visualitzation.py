from settings import *
import model
import main

def infer(dataset,
          model,
          point_cloud_file,
          shuffle_points=False,
          plot_tNet_out=True,
          return_indices_maxpool=False):
    
    '''
    This function allows to return the prediction of the class given a pointcloud.

    Parameters
    ----------
    dataset(pandas):
        The dataset that we will input.
    model(idk):
        The model that we have previously trained.
    point_cloud_file(txt):
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
    preds(idk):
        The model prediction given a pointcloud
    tnet_out(idk):
        
    '''
    num_classes = dataset.NUM_CLASSIFICATION_CLASSES
    points, label = point_cloud_file
    
    if torch.cuda.is_available():
        points = points.cuda()
        model.cuda()

    points = points.unsqueeze(dim=0)
    model = model.eval()
    preds, feature_transform, tnet_out, ix = model(points)
    preds = preds.data.max(1)[1]

    points = points.cpu().numpy().squeeze()
    preds = preds.cpu().numpy()

    if return_indices_maxpool:
        return preds, tnet_out, ix

    return preds, tnet_out




def tnet_compare(num_samples, dataset, subdataset):
    '''
    This function plots the initial pointcloud and the pointcloud represented in the canonical space (the space found by the T-Net).
    The point of the function is to have a better understanding of what the T-Net is doing.

    Parameters:
    -----------
    num_samples(int):
        The number of samples that we want to plot.
    dataset(pandas):
        The global dataset that we use to train the whole network. This is needed for the infer function.
    subdataset(pandas):
        This subdataset is the dataset where we will extract all the pointclouds samples that we want to plot.
        Usually, for the sake of rigurosity, it is used the test set.
    '''
    # Plot 7 samples
    for SAMPLE in range(num_samples):

        fig = plt.figure(figsize=[12,6]) # height and width, DO NOT CHANGE.

        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # plot input sample
        pc = subdataset[SAMPLE][0].numpy()
        label = subdataset[SAMPLE][1]
        sc = ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=pc[:,0] ,s=50, marker='o', cmap="viridis", alpha=0.7)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlim3d(-1, 1)
        ax.title.set_text(f'Input point cloud - Target: {label}')

        # plot transformation
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        preds, tnet_out = infer(dataset, model,subdataset[SAMPLE])
        points=tnet_out
        sc = ax.scatter(points[0,0,:], points[0,1,:], points[0,2,:], c=points[0,0,:] ,s=50, marker='o', cmap="viridis", alpha=0.7)
        ax.title.set_text(f'Output of "Input Transform" Detected: {preds}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(f'figures/Tnet-out-{label}.png',dpi=100)
        #print('Detected class: %s' % preds)