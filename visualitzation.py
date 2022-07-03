from settings import *
import model
from PIL import Image


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
