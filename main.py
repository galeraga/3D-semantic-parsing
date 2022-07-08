"""
PointNet implementation with S3DIS dataset
"""

#------------------------------------------------------------------------------
# IMPORTS
#------------------------------------------------------------------------------
# from numpy import double
from settings import * 
import dataset 
import model    
from tensorboardlogger import TensorBoardLogger 
from summarizer import S3DIS_Summarizer
from visualitzation import tnet_compare, tnet_compare_in_site, infer

#------------------------------------------------------------------------------
# AUX METHODS
#------------------------------------------------------------------------------
def avoid_MaxPool1d_warning(f):
    """
    Function decorator to avoid the user warning
    UserWarning: Note that order of the arguments: 
    ceil_mode and return_indices will changeto match the args list in nn.MaxPool1d in a future release.
    warnings.warn("Note that order of the arguments: ceil_mode and return_indices will change"
    """
    def function_with_warnings_removed(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print(80 * "-")
            print(task.upper()+ "ING")
            print(80 * "-")
            print("Start time: ", datetime.datetime.now())
            f(*args, **kwargs)
            print("End time: ", datetime.datetime.now())

    return function_with_warnings_removed

def task_welcome_msg(task = None):
    """
    Info message to be displayed when training/testing by epoch
    """

    msg = "Starting {}-{} with:".format(task, goal)
    msg += "\n- {} classes ({})".format(hparams['num_classes'], objects_dict)
    
    if "classification" in args.goal:
        msg += "\n- {} points per object ".format(hparams['num_points_per_object'])
    
    if "segmentation" in args.goal:
        msg += "\n- {} points per room ".format(hparams['num_points_per_room'])

    msg += "\n- {} dimensions per object ".format(hparams['dimensions_per_object'])
    msg += "\n- {} batch_size ".format(hparams['batch_size'])
    msg += "\n- {} workers ".format(hparams['num_workers'])
    
    if hparams['device'] == "cpu":
        msg += "\n- device: {}".format(hparams['device'])
    else:
        msg += "\n- device: {} ({}x {})".format(hparams['device'],
                                        torch.cuda.device_count(),
                                        torch.cuda.get_device_name(0)
                                        )

    print(msg)

def create_dataloaders(ds):
    """
    Creates the dataloaders
    """

    train_dataset = ds[0]
    val_dataset = ds[1]
    test_dataset = ds[2]

    # Dataloaders creation
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size = hparams['batch_size'], 
            shuffle = True,
            num_workers = hparams["num_workers"]
            )
    
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = hparams['batch_size'], 
            shuffle = True,
            num_workers = hparams["num_workers"]
            )
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size = hparams['batch_size'], 
            shuffle = True,
            num_workers = hparams["num_workers"]
            )
    
    return train_dataloader, val_dataloader, test_dataloader


#------------------------------------------------------------------------------
# CLASSIFICATION METHODS
#------------------------------------------------------------------------------
@avoid_MaxPool1d_warning
def train_classification(model, dataloaders):
    """
    Train the PointNet network for classification goals

    Inputs:
        - model: the PointNet model class
        - dataloaders: train, val and test

    Outputs:
        - None
    """

    # Task welcome message
    task_welcome_msg(task = "train")
    
    # Get the proper dataloaders
    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]

    # Aux vars for grand totals
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_loss= np.inf
    total_train_time = []
    total_val_time = []
    
    optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

    for epoch in range(1, hparams['epochs'] + 1):
        # Aux vars per epoch
        epoch_train_loss = []
        epoch_train_acc = []
        tnet_out_list = []
        epoch_train_start_time = datetime.datetime.now()

        tqdm_desc = "{}ing epoch ({}/{})".format(task.capitalize(), epoch, hparams['epochs'])
        # training loop
        for data in tqdm(train_dataloader, desc = tqdm_desc):
            model = model.train()
            
            points, targets = data  
            
            points = points.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            preds, feature_transform, tnet_out, ix_maxpool = model(points)           

            # Why?  
            identity = torch.eye(feature_transform.shape[-1]).to(device)

            # Formula (2) in original paper (Lreg)
            # TODO: According to the original paper, it should only be applied
            # during the alignment of the feature space (with higher dimension (64))
            # than the spatial transformation matrix (3)
            # With the regularization loss, the model optimization becomes more
            # stable and achieves better performance
            # A regularization loss (with weight 0.001) is added to the softmax
            # classification loss to make the matrix close to ortoghonal
            # (quoted from supplementary info from the original paper)
            
            regularization_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
            
            # Loss: The negative log likelihood loss 
            # It is useful to train a classification problem with C classes.
            # torch.nn.functional.nll_loss(input, target, ...) 
            # ----For classification:
            # input – (N,C) (N: batch_size; C: num_classes) 
            # target - (N)
            # preds.shape[batch_size, num_classes]
            # targets.shape[batch_size], but every item in the target tensor
            # must be in the range of num_classes - 1 
            # E.g: if num_classes = 2 -> target[i] < 2 {0, 1}
            
            # Why does the loss function return a single scalar for a batch?
            # It returns, by default, the weighted mean of the output 
            targets = targets.squeeze(dim = -1)
            loss = F.nll_loss(preds, targets.long()) + 0.001 * regularization_loss
            
            epoch_train_loss.append(loss.cpu().item())
        
            loss.backward()
            optimizer.step()
            
            # From the num_classes dimension (dim =1), find out the max value
            # max() returns a tuple (max_value, idx_of_the_max_value)
            # Take the index of the max value, since the object class 
            # classification is based on the position of the max value
            # https://pytorch.org/docs/stable/generated/torch.max.html
            # Similar to torch.argmax, that returns the second value
            # returned by torch.max()
            preds = preds.data.max(dim = 1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            accuracy = corrects.item() / preds.numel()
            epoch_train_acc.append(accuracy)
            tnet_out_list.append(tnet_out)

            
        epoch_train_end_time = datetime.datetime.now()
        train_time_per_epoch = (epoch_train_end_time - epoch_train_start_time).seconds
        total_train_time.append(train_time_per_epoch)

        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_start_time = datetime.datetime.now()

        tqdm_desc = "{} epoch ({}/{})".format("Validating", epoch, hparams['epochs'])
        # validation loop
        for data in tqdm(val_dataloader, desc = tqdm_desc):
            model = model.eval()
    
            points, targets = data
            points = points.to(device)
            targets = targets.to(device)
                    
            preds, feature_transform, tnet_out, ix = model(points)
            
            # loss = F.nll_loss(preds, targets)
            targets = targets.squeeze(dim = -1)
            loss = F.nll_loss(preds, targets.long()) + 0.001 * regularization_loss
            
            
            epoch_val_loss.append(loss.cpu().item())
            
            preds = preds.data.max(dim = 1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            accuracy = corrects.item() / preds.numel()
            epoch_val_acc.append(accuracy)

        epoch_val_end_time = datetime.datetime.now()
        val_time_per_epoch = (epoch_val_end_time - epoch_val_start_time).seconds
        total_val_time.append(val_time_per_epoch)

        print('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f, time(secs): %s'
                % (epoch,
                    round(np.mean(epoch_train_loss), 4),
                    round(np.mean(epoch_val_loss), 4),
                    round(np.mean(epoch_train_acc), 4),
                    round(np.mean(epoch_val_acc), 4),
                    train_time_per_epoch + val_time_per_epoch
                    )
                )

        if np.mean(val_loss) < best_loss:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(
                state, 
                os.path.join(eparams['pc_data_path'], 
                    eparams["checkpoints_folder"],
                    eparams["checkpoint_name"])
                )
            best_loss=np.mean(val_loss)

        train_loss.append(np.mean(epoch_train_loss))
        val_loss.append(np.mean(epoch_val_loss))
        train_acc.append(np.mean(epoch_train_acc))
        val_acc.append(np.mean(epoch_val_acc))

        # Log results to TensorBoard for every epoch
        logger.writer.add_scalar(goal.capitalize() + " Loss/Training", train_loss[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Loss/Validation", val_loss[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Accuracy/Training", train_acc[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Accuracy/Validation", val_acc[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Time/Training", total_train_time[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Time/Validation", total_val_time[-1], epoch)
    
    print("Mean accuracy: Training: {:.4f} | Validation: {:.4f} ".format(
            sum(train_acc)/len(train_acc),
            sum(val_acc)/len(val_acc),
            ))
    print("Total time (seconds) for {}ing {}: {} secs ".format( 
                    task,
                    goal,
                    sum(total_train_time) + sum(total_val_time))
                    )

@avoid_MaxPool1d_warning
@torch.no_grad()
def test_classification(model, dataloaders):
    """
    Test the PointNet classification network
    """

    # Task welcome message
    task_welcome_msg(task = "test")
    
    # Path to the checkpoint file
    model_checkpoint = os.path.join(
            eparams['pc_data_path'], 
            eparams['checkpoints_folder'], 
            eparams["checkpoint_name"]
            )
    
    # If the checkpoint does not exist, train the model
    if not os.path.exists(model_checkpoint):
        print("The model does not seem already trained! Starting the training rigth now from scratch...")
        train_classification(model, dataloaders)
    
    # Loading the existing checkpoint
    print("Loading checkpoint {} ...".format(model_checkpoint))
    state = torch.load(
                model_checkpoint, 
                map_location = torch.device(hparams["device"]))
    model.load_state_dict(state['model'])  

    # Select the proper dataloader
    test_dataloader = dataloaders[2]

    # Aux test vars
    accuracies = []
    
    # Enter evaluation mode
    model.eval()
    
    # Test the model
    print("Testing data classification")
    for batch_idx, data in enumerate(tqdm(test_dataloader)):
        points, target_labels = data        
                
        points = points.to(device)
        target_labels = target_labels.to(device)
        
        preds, feature_transform, tnet_out, ix = model(points)
        
        # preds.shape([batch_size, num_classes])
        preds = preds.data.max(1)[1]
        
        corrects = preds.eq(target_labels.data).cpu().sum()
        accuracy = corrects.item() / preds.numel()
        accuracies.append(accuracy)
        
        logger.writer.add_scalar(goal.capitalize() + " Accuracy/Test", accuracy, batch_idx)
    
    mean_accuracy = (torch.FloatTensor(accuracies).sum()/len(accuracies))
    print("Average accuracy: {:.2f} ".format(float(mean_accuracy)))
               
#------------------------------------------------------------------------------
# SEGMENTATION METHODS
#------------------------------------------------------------------------------
@avoid_MaxPool1d_warning
def train_segmentation(model, dataloaders):
    """
    Train the PointNet network for semantic segmentation tasks

    Inputs:
        - model: the PointNet model class
        - dataloaders: train, val and test

    Outputs:
        - None
    """

    # Task welcome message
    task_welcome_msg(task = "train")
    
    # Get the proper dataloaders
    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]

    # Aux vars for grand totals
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_loss= np.inf
    total_train_time = []
    total_val_time = []

    # Per object accuracy
    per_object_accuracy = dict()
    
    optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

    for epoch in range(1, hparams['epochs'] +1):
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_train_start_time = datetime.datetime.now()

        for k,v in objects_dict.items():
            per_object_accuracy[k] = []
            

        tqdm_desc = "{}ing epoch ({}/{})".format(task.capitalize(), epoch, hparams['epochs'])
        # training loop
        for data in tqdm(train_dataloader, desc = tqdm_desc):
            model = model.train()
            
            points, targets = data  
            points = points.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            preds, feature_transform, tnet_out, ix_maxpool = model(points)

            # Why?  
            identity = torch.eye(feature_transform.shape[-1]).to(device)

            # Formula (2) in original paper (Lreg)
            # TODO: According to the original paper, it should only be applied
            # during the alignment of the feature space (with higher dimension (64))
            # than the spatial transformation matrix (3)
            # With the regularization loss, the model optimization becomes more
            # stable and achieves better performance
            # A regularization loss (with weight 0.001) is added to the softmax
            # classification loss to make the matrix close to ortoghonal
            # (quoted from supplementary info from the original paper)    
        
            regularization_loss = torch.norm(
                identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
            
            # Loss: The negative log likelihood loss 
            # It is useful to train a classification problem with C classes.
            # torch.nn.functional.nll_loss(input, target, ...) 
            # ----For segmentation:
            # (N: batch_size; C: num_classes) 
            # input (predictions)– (N,C, d1, d2,...,dk)) 
            # target - (N, d1, d2,...,dk)
            # So shapes have to be:
            # preds.shape[batch_size, num_classes, max_points_per_room]
            
            # Why does the loss function return a single scalar for a batch?
            # It returns, by default, the weighted mean of the output 
            # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
            loss = F.nll_loss(preds, targets.long()) + 0.001 * regularization_loss

            # epoch_train_loss.append(loss.cpu().item())
            epoch_train_loss.append(loss.cpu().item())
        
            loss.backward()
            
            optimizer.step()
            
            # From the num_classes dimension (dim =1), find out the max value
            # max() returns a tuple (max_value, idx_of_the_max_value)
            # Take the index of the max value, since the object class 
            # classification is based on the position of the max value
            # https://pytorch.org/docs/stable/generated/torch.max.html
            preds = preds.data.max(dim = 1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            accuracy = corrects.item() / preds.numel()
            epoch_train_acc.append(accuracy)

            for k,v in objects_dict.items():
                # From the label tensor, get only the ones of that object
                annotated_labels_per_object = targets.eq(v).cpu().sum()
                predicted_labels_per_object = preds.eq(v).cpu().sum()
                per_object_accuracy[k].append((annotated_labels_per_object, predicted_labels_per_object))
            
        epoch_train_end_time = datetime.datetime.now()
        train_time_per_epoch = (epoch_train_end_time - epoch_train_start_time).seconds
        total_train_time.append(train_time_per_epoch)

        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_start_time = datetime.datetime.now()

        tqdm_desc = "{} epoch ({}/{})".format("Validating", epoch, hparams['epochs'])
        # validation loop
        for data in tqdm(val_dataloader, desc = tqdm_desc):
            model = model.eval()
    
            points, targets = data
            points = points.to(device)
            targets = targets.to(device)
                    
            preds, feature_transform, tnet_out, ix = model(points)

            loss = F.nll_loss(preds, targets.long()) + 0.001 * regularization_loss
        
            epoch_val_loss.append(loss.cpu().item())
            
            preds = preds.data.max(dim = 1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            accuracy = corrects.item() / preds.numel()       
            epoch_val_acc.append(accuracy)

        epoch_val_end_time = datetime.datetime.now()
        val_time_per_epoch = (epoch_val_end_time - epoch_val_start_time).seconds
        total_val_time.append(val_time_per_epoch)

        print('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f, time(secs): %s'
            % (epoch,
                round(np.mean(epoch_train_loss), 4),
                round(np.mean(epoch_val_loss), 4),
                round(np.mean(epoch_train_acc), 4),
                round(np.mean(epoch_val_acc), 4),
                train_time_per_epoch + val_time_per_epoch
                )
            )

        msg = 'Per object points (Annotated | Predicted): \n'
        for k,v in per_object_accuracy.items():
            total_annotated = sum(a.item() for a,p in v)
            total_predicted = sum(p.item() for a,p in v)
            msg += "- {}: ({}|{}) ({:.2f}%) \n".format(k, total_annotated, 
                    total_predicted, (total_predicted/total_annotated)*100)
            
            tb_desc = goal.capitalize() + " Accuracy Per Object " + "(" + task + ")/" + k
            logger.writer.add_scalars(tb_desc, 
                    {"Annotated": total_annotated, 
                    "Predicted": total_predicted}, 
                    epoch)     
        print(msg)
        
        if np.mean(val_loss) < best_loss:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(
                state, 
                os.path.join(eparams['pc_data_path'], 
                    eparams["checkpoints_folder"],
                    eparams["checkpoint_name"])
                )
            best_loss=np.mean(val_loss)

        train_loss.append(np.mean(epoch_train_loss))
        val_loss.append(np.mean(epoch_val_loss))
        train_acc.append(np.mean(epoch_train_acc))
        val_acc.append(np.mean(epoch_val_acc))

        # Log results to TensorBoard for every epoch
        logger.writer.add_scalar(goal.capitalize() + " Loss/Training", train_loss[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Loss/Validation", val_loss[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Accuracy/Training", train_acc[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Accuracy/Validation", val_acc[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Time/Training", total_train_time[-1], epoch)
        logger.writer.add_scalar(goal.capitalize() + " Time/Validation", total_val_time[-1], epoch)
    
    print("Mean accuracy: Training: {:.4f} | Validation: {:.4f} ".format(
            sum(train_acc)/len(train_acc),
            sum(val_acc)/len(val_acc),
            ))
    print("Total time (seconds) for {}ing {}: {} secs ".format( 
                task,
                goal,
                sum(total_train_time) + sum(total_val_time))
                )

@avoid_MaxPool1d_warning
@torch.no_grad()
def test_segmentation(model, dataloaders):
    """
    Test the PointNet segmenatation network
    """

    # Task welcome message
    task_welcome_msg(task = "test")
    
    # Path to the checkpoint file
    model_checkpoint = os.path.join(
            eparams['pc_data_path'], 
            eparams['checkpoints_folder'], 
            eparams["checkpoint_name"]
            )
    
    # If the checkpoint does not exist, train the model
    if not os.path.exists(model_checkpoint):
        print("The model does not seem already trained! Starting the training rigth now from scratch...")
        train_classification(model, dataloaders)
    
    # Loading the existing checkpoint
    print("Loading checkpoint {} ...".format(model_checkpoint))
    state = torch.load(
                model_checkpoint, 
                map_location = torch.device(hparams["device"]))
    model.load_state_dict(state['model'])  

    # Select the proper dataloader
    test_dataloader = dataloaders[2]

    # Aux test vars
    accuracies = []
    
    # Enter evaluation mode
    model.eval()
    
    # Test the model
    print("Testing data segmentation")
    for batch_idx, data in enumerate(tqdm(test_dataloader)):
        points, target_labels = data        
                
        points = points.to(device)
        target_labels = target_labels.to(device)
        
        preds, feature_transform, tnet_out, ix = model(points)
        
        # preds.shape([batch_size, num_classes])
        preds = preds.data.max(1)[1]
        
        corrects = preds.eq(target_labels.data).cpu().sum()
        accuracy = corrects.item() / preds.numel()
        accuracies.append(accuracy)
        
        logger.writer.add_scalar(goal.capitalize() + " Accuracy/Test", accuracy, batch_idx)
    
    mean_accuracy = (torch.FloatTensor(accuracies).sum()/len(accuracies))*100
    print(80 * "-")
    print("Average accuracy: {:.2f}%".format(float(mean_accuracy)))
    print(80 * "-")


@avoid_MaxPool1d_warning
@torch.no_grad()    
def watch_segmentation(model, dataloaders, random = False):
    """
    Visualize how PointNet segments objects in a single room.

    All the points of a single room are taken for visualization in order to have
    a visually smooth representation of the room.

    Since all the points of the rooms are going to be taken, no dataloaders can 
    be used since dataloaders return an smaller amount of points per room/sliding 
    window due to their sampling process.

    At least, two ways can be folloed to achieve this goal:
    1.- Read directly the annotated file (e.g., Area_6_office_33_annotated.txt)
    2.- Read from sliding windows (e.g., Area_6_office_33_win14.pt)

    The latter option is prefered in order to have the ability to also display 
    per sliding window information, if desired.

    Workflow overview:
    1.- Pick randomly one of the available sliding windows
    2.- Get the Area_N Space_X from this randomly selected sliding window
    3.- Get all the sliding windows from Area_N Space_X
    4.- Get object info per sliding window

    Output: 
    A dict containing:
    - As keys: The sliding window ID (winID) 
    - As values: A list of list. Every list with:
        - object name (chair, table,...)
        - object_ID (from the proper dict)
        - amount of annotated points this object has in this room (as a tensor)
        - amount of predicted points for this object in this room (as a tensor)
        - the actual predicted points, in relative coordinates (as a tensor)
        - the actual predicted points, in absolute coordinates (as a tensor)
    """
    
    print("Visualizing data segmentation")
    
    # Enter evaluation mode
    model.eval()

    # Path to the checkpoint file
    model_checkpoint = os.path.join(
            eparams['pc_data_path'], 
            eparams['checkpoints_folder'], 
            eparams["checkpoint_name"]
            )
    
    # If the checkpoint does not exist, train the model
    if not os.path.exists(model_checkpoint):
        print("The model does not seem already trained! Starting the training rigth now from scratch...")
        train_classification(model, dataloaders)
    
    # Loading the existing checkpoint
    print("Loading checkpoint {} ...".format(model_checkpoint))
    state = torch.load(
                model_checkpoint, 
                map_location = torch.device(hparams["device"]))
    model.load_state_dict(state['model'])  
  
    # Select the object to detect
    # segmentation_target_object is defined in settings.py
    # Get the ID from the proper dic (either "all", "movable" or "structural")
    # From the summary file, these are the available dicts:
    # 'all': {'ceiling': 0, 'clutter': 1, 'door': 2, 'floor': 3, 'wall': 4, 'beam': 5, 'board': 6, 'bookcase': 7, 'chair': 8, 'table': 9, 'column': 10, 'sofa': 11, 'window': 12, 'stairs': 13}, 
    # 'movable': {'clutter': 0, 'board': 1, 'bookcase': 2, 'chair': 3, 'table': 4, 'sofa': 5}, 
    # 'structural': {'ceiling': 0, 'clutter': 1, 'door': 2, 'floor': 3, 'wall': 4, 'beam': 5, 'column': 6, 'window': 7, 'stairs': 8}}
    dict_to_use = objects_dict

    # Select a random sliding window from an office room 
    # (e.g, 'Area_6_office_33_win14.pt')
    if random:
        picked_sliding_window = random.choice([i for i in test_ds.sliding_windows if "office" in i])
        area_and_office ='_'.join(picked_sliding_window.split('_')[0:4])
    else:
        area_and_office = target_room_for_visualization
    
    print("Randomly selected office to plot: ", area_and_office)
    # To avoid getting office11, office12, when the randomly selectd office is office_1 
    area_and_office += "_"

    # Get all the sliding windows related to the picked one
    # to have a single room
    # (e.g. 'Area_6_office_33_win0.pt', 'Area_6_office_33_win1.pt',...)
    all_sliding_windows_for_a_room = sorted([i for i in test_ds.sliding_windows if area_and_office in i]) 
    
    # Load the tensors from the files
    room_tensors = []
    for f in all_sliding_windows_for_a_room:
        # Get the name of the WinID. We'll be used didct keys
        winID = f.split('_')[-1].split('.')[0]    
        
        path_to_sliding_window_file = os.path.join(
                path_to_current_sliding_windows_folder, 
                f)
        room_tensors.append(
            (torch.load(path_to_sliding_window_file, map_location = torch.device(hparams["device"])),
            winID
            ))

    # Define the process bar to display when processing files
    progress_bar = tqdm(room_tensors, total = len(room_tensors))
       
    # data = torch.cat(room_tensors, dim = 0)
    # Define the output dict containing the points to display per WinID
    out_dict = dict()

    for (data, win_id) in progress_bar:

        msg = "{} - Splitting points".format(win_id)    
        progress_bar.set_description(msg)

        # The amount of cols to return per room will depend on whether or not
        # color must be taken into account when data is fed into the model
        # room -> [x_rel y_rel z_rel r g b x_abs y_abs x_abs winID label] (11 cols)
        points_rel = data[:, :hparams["dimensions_per_object"]].to(device)
        points_color = data[:, 3:6].to(device)
        points_abs = data[:, -3:].to(device)
        target_labels = data[:, -1].to(device)

        # Translate labels to the proper dict!
       # msg = "{} - Translating labels (with {} points)".format(win_id, len(target_labels))    
       # progress_bar.set_description(msg)
       # target_labels = dataset.AdaptNumClasses(target_labels, all_dicts).adapt()
       
        # From all the points in the room, find out how many of them belong to 
        # the different objects
        # We're going to get per object info (a list of lists) containing:
        # - object, 
        # - object_ID, 
        # - amount of annotated points this object has in this room (as a tensor)
        # - amount of predicted points for this object in this room (as a tensor) (initialized to zero)]
        # - predicted points, in relative coordinates (as a tensor)
        # - predicted points, in absolute coordinates (as a tensor)
        point_breakdown = []
        for k,v in dict_to_use.items():
            point_breakdown.append([k, v, target_labels.eq(v).cpu().sum(), torch.tensor([0]), None, None])        
        
        # Work with points_rel (instead of points_abs)
        # Unsquezze the data tensor to give it the depth of batch_size = 1,
        # since we're going to process a single room only
        points = points_rel.unsqueeze(dim = 0)
        
        # Test the model
        # Model input: points.shape([batch_size, room_points, dimensons_per_point)]
        # Model output: preds.shape([batch_size, num_classes, room_points])
        msg = "{} - Feeding the model (with {} points)".format(win_id, len(points_rel))    
        progress_bar.set_description(msg)
        preds, feature_transform, tnet_out, ix = model(points)

        # Output after argmax: preds.shape([batch_size, room_points])
        preds = preds.data.max(1)[1]

        msg = "{} - Saving predictions".format(win_id)    
        progress_bar.set_description(msg)

        # Save predictions per object
        for i in point_breakdown:
            # Select the object_id of the element to check accuraracy
            id = i[1]

            # Save predictions for that object
            preds = torch.squeeze(preds, dim = 0)
            i[3] = preds.eq(id).cpu().sum()

            # Get the points identified as target objects    
            # Get the indices of preds that match the object_id
            # From preds after argmax, get the indexes in dim = 1 that match
            # the object class (segmentation_target_object_id) we want to display
            # These indexes in dim = 1 in preds (after argmax) should be a mapping 
            # of the the indexes in dim = 1 in points
            # Tricky method:
            # - torch.where() returns 1. where condition is met, 0. elsewhere 
            # - torch.nonzero() returns a tensor containing the indices of all non-zero elements of ones_mask
            # - torch.index_select() returns a new tensor which indexes the input tensor along 
            #   dimension dim using the entries in indices
            ones_mask = torch.where(preds == id, 1., 0.).squeeze(dim = 0)
            indices = torch.nonzero(ones_mask).squeeze(dim = 1)
            # points = points.squeeze(dim = 0)

            # Save the points to display
            # Points to display (relative coordinates)
            i[4] = torch.index_select(points_rel, 0, indices)
            # Points to display (absolute coordinates)
            i[5] =  torch.index_select(points_abs, 0, indices)
    
    
        # Save the results in a dict
        out_dict[win_id] = point_breakdown
      
    # Confussion Matrix (nested dict)
    # True positives (tp)
    # False positives (fp)
    # True negatives (tn)
    # False negatives (fn)
    available_values = ("tp", "fp", "tn", "fn")
    confussion_matrix_dict = dict()

    for k in objects_dict:
        confussion_matrix_dict[k] = {}
        for i in available_values:
            confussion_matrix_dict[k][i] = []

       
    for k, v in out_dict.items():
        print(80 * "-")
        print("WinID: ", k)
        for i in v:
            print(i)            
            # Analyze results
            # The difference between predicted and annotated points
            delta_points = abs(i[3] - i[2])
            # True Positives
            if (i[2].item() != 0) and (i[3].item() != 0):
                confussion_matrix_dict[i[0]]["tp"].append(i[2])
                if i[3] >= i[2]:
                    confussion_matrix_dict[i[0]]["fp"].append(delta_points)
                else:
                    confussion_matrix_dict[i[0]]["fn"].append(delta_points)
            # True Negatives
            elif (i[2].item() == 0) and (i[3].item() == 0):
                confussion_matrix_dict[i[0]]["tn"].append(delta_points)
            # False Positives
            elif (i[2].item() == 0) and (i[3].item() != 0):
                confussion_matrix_dict[i[0]]["fp"].append(delta_points)
            # False Negatives
            elif (i[2].item() != 0) and (i[3].item() == 0):
                confussion_matrix_dict[i[0]]["fn"].append(delta_points)

    print(80 * "-")
    print("Values for confussion matrix for {}".format(model_checkpoint))
    for k,v in confussion_matrix_dict.items():
        print(20 * "-")
        print("{}: ".format(k))
        for par, val in v.items():           
            print("\t{}: {}".format(par, sum(val)))
            
        tp = 0 if len(v["tp"]) == 0 else sum(v["tp"]).item() 
        tn = 0 if len(v["tn"]) == 0 else sum(v["tn"]).item() 
        fp = 0 if len(v["fp"]) == 0 else sum(v["fp"]).item() 
        fn = 0 if len(v["fn"]) == 0 else sum(v["fn"]).item() 

        # When true positive + false positive == 0, precision is undefined. 
        # When true positive + false negative == 0, recall is undefined. 
        try:
            accuracy = ((tp + tn) / (tp + tn + fp + fn))*100
            accuracy = "{:.2f} %".format(accuracy)
        except ZeroDivisionError:
            accuracy = "N/A"
        try:    
            precision = (tp/(tp + fp))*100
            precision = "{:.2f} %".format(precision)
        except ZeroDivisionError: 
            precision = "N/A (tp + fp = 0)"
        try:
            recall = (tp/(tp + fn))*100
            recall = "{:.2f} %".format(recall)
        except ZeroDivisionError:
            recall = "N/A (tp + fn = 0)"

        print("\tAccuracy: {}".format(accuracy))
        print("\tPrecision: {}".format(precision))
        print("\tRecall: {}".format(recall))
               
    
    # TODO: Insert Lluis' code here for visualization
    # out_dict contains all the points detected for all objects
    # lluis_code(data, segmentation_target_object_id, points_to_display) 


#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------
if __name__ == "__main__":

    # When choices are given in parser add_argument, 
    # the parser returns a list 
    # goal -> either "classification" or "segmentation"
    # task -> either "train" or "test"
    goal = ''.join(args.goal)
    task = ''.join(args.task)
    
    # Prepare to run on CUDA/CPU
    device = hparams['device']

    # Create a TensorBoard logger instance
    logger = TensorBoardLogger(args)

    # Create the ground truth file for classification
    summary_file = S3DIS_Summarizer(eparams["pc_data_path"], logger)

    # Get the dicts:
    objects_dict = summary_file.get_labels()

    # Create the ground truth files for semantic segmentation
    if "segmentation" in args.goal:
        summary_file.label_points_for_semantic_segmentation()
        summary_file.create_sliding_windows()

    # Log insights from the S3DIS dataset into TensorBoard
    logger.log_dataset_stats(summary_file)

    # Logging hparams for future reference
    logger.log_hparams(hparams)
    
    # Define the checkpoint name
    eparams["checkpoint_name"] = "S3DIS_checkpoint_{}_{}_points_{}_dims_{}_num_classes_{}_epochs.pth".format(
                                            goal,
                                            hparams["num_points_per_object"] if goal == "classification" else hparams["num_points_per_room"],
                                            hparams["dimensions_per_object"],
                                            hparams["num_classes"],
                                            hparams["epochs"],
                                            )
    
    # Dataset instance creation (goal-dependent) 
    # If goal == classification: 
    #   - S3DISDataset4ClassificationTrain, 
    #   - S3DISDataset4ClassificationVal
    #   - S3DISDataset4ClassificationTest 
    # If goal == segmentation: 
    #   - S3DISDataset4SegmentationTrain
    #   - S3DISDataset4ClassificationVal
    #   - S3DISDataset4ClassificationTest
    train_ds_to_call = "S3DISDataset4" + goal.capitalize()  + "Train"
    val_ds_to_call = "S3DISDataset4" + goal.capitalize()  + "Val"
    test_ds_to_call = "S3DISDataset4" + goal.capitalize()  + "Test"
    
    train_ds = getattr(dataset, train_ds_to_call)(eparams['pc_data_path'], objects_dict, transform = None)
    val_ds = getattr(dataset, val_ds_to_call)(eparams['pc_data_path'], objects_dict, transform = None)
    test_ds = getattr(dataset, test_ds_to_call)(eparams['pc_data_path'], objects_dict, transform = None)
    
    ds = train_ds, val_ds, test_ds    
    
    # Show info about the ds
    for i in ds:
        print(i)
    
    # Create the dataloaders
    # dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    dataloaders = create_dataloaders(ds)

    # Model instance creation (goal-dependent)
    # If goal == classification -> ClassificationPointNet
    # If goal == segmentation -> SegmentationPointNet
    model_to_call = goal.capitalize() + "PointNet"
    model = getattr(model, model_to_call)(num_classes = hparams['num_classes'],
                                   point_dimension = hparams['dimensions_per_object']).to(device)

    # Print info about the model with torchinfo
    # summary(model, input_size=(hparams['batch_size'], hparams['max_points_per_space'], hparams['dimensions_per_object']))

    # Carry out the the task to do
    # (e.g, train_classification(), test_segmentation(), watch_segmentation())
    
    locals()[task + "_" + goal](model, dataloaders)
    
    # Close TensorBoard logger and send runs to TensorBoard.dev
    logger.finish()
    #tnet_compare(model, ds)


