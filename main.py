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
            print("Start time: ", datetime.datetime.now())
            f(*args, **kwargs)
            print("End time: ", datetime.datetime.now())

    return function_with_warnings_removed

def task_welcome_msg(task = None):
    """
    Info message to be displayed when training/testing by epoch
    """

    msg = "Starting {}-{} with:".format(task, goal)
    msg += "\n- {} classes ({})".format(hparams['num_classes'], all_dicts[''.join(args.objects)])
    
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

    optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

    for epoch in range(1, hparams['epochs'] +1):
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_train_start_time = datetime.datetime.now()

        tqdm_desc = "{}ing epoch ({}/{})".format(task.capitalize(), epoch, hparams['epochs'])
        # training loop
        for data in tqdm(train_dataloader, desc = tqdm_desc):
            model = model.train()
            
            # TODO: Insert Clara's code here
            # split_the_room_into_blocks(data) -> list_of_blocks
            # for each block in list_of_block:
            #   points, targets = block
            #   points = points.to(device)
            #   targets = targets.to(device)
            #   optimizer.zero_grad()
            #   model(block)
            #   Move the rest of the code (accuracy, etc) within the for loop
            
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
            
            # TODO: loss has to be defined for semantic segmentation   
            # TODO: Is needed the same regularization loss that classification??
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
def visualize_segmentation_short(model):
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

    Workflow:
    1.- Pick randomly one of the available sliding windows
    2.- Get the Area_N Space_X from this randomly selected sliding window
    3.- Get all the sliding windows from Area_N Space_X
    4.- Join all the sliding windows into a single (torch) file    
    """
    
    print("Visualizing data segmentation")
    
    # Enter evaluation mode
    model.eval()

    # Aux test vars
    accuracies = []
    
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
    dict_to_use = all_dicts[''.join(args.objects)]
    segmentation_target_object_id = dict_to_use[segmentation_target_object]

    # Select a random sliding window from an office room 
    # (e.g, 'Area_6_office_33_win14.pt')
    picked_sliding_window = random.choice([i for i in test_ds.sliding_windows if "office" in i])
    area_and_office ='_'.join(picked_sliding_window.split('_')[0:4])
    
    # Get all the sliding windows related to the picked one
    # to have a single room
    # (e.g. 'Area_6_office_33_win0.pt', 'Area_6_office_33_win1.pt',...)
    all_sliding_windows_for_a_room = [i for i in test_ds.sliding_windows if area_and_office in i] 
    
    room_tensors = []
    for f in all_sliding_windows_for_a_room:
        path_to_sliding_window_file = os.path.join(
                path_to_current_sliding_windows_folder, 
                f)
        room_tensors.append(
            torch.load(path_to_sliding_window_file, map_location = torch.device(hparams["device"])))

    data = torch.cat(room_tensors, dim = 0)

    # The amount of cols to return per room will depend on whether or not
    # color must be taken into account when data is fed into the model
    # room -> [x_rel y_rel z_rel r g b x_abs y_abs x_abs winID label] (11 cols)
    print("Getting data and labels")  
    points_rel = data[:, :hparams["dimensions_per_object"]].to(device)
    points_color = data[:, 3:6].to(device)
    points_abs = data[:, -3:].to(device)
    target_labels = data[:, -1].to(device)

    # Translate labels to the proper dict!
    print("Max:", torch.max(target_labels))
    target_labels = dataset.AdaptNumClasses(target_labels, all_dicts).adapt()
    print("Max:", torch.max(target_labels))
    


    # From all the points in the room, find out how many of them belong to 
    # the different objects
    total_points_annotated_as_target_id = target_labels.eq(segmentation_target_object_id).cpu().sum() 
    print("Randomly selected room to visualize: {} (Total points: {})".format(area_and_office, len(data)))
    print("Object to detect: {0} (ID:{1}) (Total annotated {0} points: {2})".format(
        segmentation_target_object, 
        segmentation_target_object_id, 
        total_points_annotated_as_target_id))
    
    if total_points_annotated_as_target_id.item() == 0:
        print("There're no {}s in room {}!".format(segmentation_target_object, area_and_office))
        return
    
    else: 
        # Find out per object info
        # A list of lists:
        # [object, object_ID, 
        # amount of annotated points this object has in this room (as a tensor)
        # amount of predicted points for this object in this room (as a tensor) (initialized to zero)]
        point_breakdown = []
        for k,v in dict_to_use.items():
            point_breakdown.append([k, v, target_labels.eq(v).cpu().sum(), torch.tensor([0])])        
        
        # Work with points_rel (instead of points_abs)
        # Unsquezze the data tensor to give it the depth of batch_size = 1,
        # since we're going to process a single room only
        points = points_rel.unsqueeze(dim = 0)
        
        # Test the model
        # Model input: points.shape([batch_size, room_points, dimensons_per_point)]
        # Model output: preds.shape([batch_size, num_classes, room_points])
        preds, feature_transform, tnet_out, ix = model(points)

        # Output after argmax: preds.shape([batch_size, room_points])
        preds = preds.data.max(1)[1]

        # Save predictions per object
        for i in point_breakdown:
            # Select the object_id of the element to check accuraracy
            id = i[1]
            # Save predictions for that object
            i[3] = preds.eq(id).cpu().sum()
       
        print(80 * "-")
        print("Model performance (annotated | predicted) points per object:")
        print(80 * "-")
        for obj, id, qty, qty_pred in point_breakdown:
            print("{} (ID:{}): {} | {}".format(obj, id, qty.item(), qty_pred.item()))
        

        #corrects = preds.eq(target_labels.data).cpu().sum()
        #accuracy = corrects.item() / total_points_annotated_as_target_id
        #accuracies.append(accuracy)
        
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
        ones_mask = torch.where(preds == segmentation_target_object_id, 1., 0.).squeeze(dim = 0)
        indices = torch.nonzero(ones_mask).squeeze(dim = 1)
        # points = points.squeeze(dim = 0)
        points_to_display = torch.index_select(points_abs, 0, indices)

        # TODO: Insert Lluis' code here for visualization
        # points is the whole room points
        # lluis_code(data, segmentation_target_object_id, points_to_display) 
        
        
        #logger.writer.add_scalar(goal.capitalize() + " Accuracy/Visualization", accuracy)

        #mean_accuracy = (torch.FloatTensor(accuracies).sum()/len(accuracies))*100
        #print(80 * "-")
        #print("Average accuracy: {:.2f}%".format(float(mean_accuracy)))
        #print(80 * "-")

@avoid_MaxPool1d_warning
@torch.no_grad()  
def visualize_segmentation_long(model):
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

    Workflow:
    1.- Pick randomly one of the available sliding windows
    2.- Get the Area_N Space_X from this randomly selected sliding window
    3.- Get all the sliding windows from Area_N Space_X
    4.- Join all the sliding windows into a single (torch) file    
    """
    
    print("Visualizing data segmentation")
    
    # Enter evaluation mode
    model.eval()

    # Aux test vars
    accuracies = []
    
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
    dict_to_use = all_dicts[''.join(args.objects)]
    segmentation_target_object_id = dict_to_use[segmentation_target_object]

    # Select a random sliding window from an office room 
    # (e.g, 'Area_6_office_33_win14.pt')
    picked_sliding_window = random.choice([i for i in test_ds.sliding_windows if "office" in i])
    area = picked_sliding_window.split('_')[0] + '_' + picked_sliding_window.split('_')[1]
    room = picked_sliding_window.split('_')[2] + '_' + picked_sliding_window.split('_')[3]
    area_and_room ='_'.join(picked_sliding_window.split('_')[0:4])
    
    path_to_room_file = os.path.join(eparams['pc_data_path'], area, room, room + "_annotated.txt")
    print("Reading room X file from CSV to NumPy array")
    data = np.genfromtxt(path_to_room_file, 
                dtype = float, 
                skip_header = 1, 
                delimiter = '', 
                names = None) 
    print("Converting NumPy array to Pytorch tensor")
    data = torch.from_numpy(data).float()


    # The amount of cols to return per room will depend on whether or not
    # color must be taken into account when data is fed into the model
    # room -> [x y z r g b label] (7 cols)
    print("Getting data and labels")  
    points = data[:, :hparams["dimensions_per_object"]].to(device)
    target_labels = data[:, -1].to(device)

    # Translate labels to the proper dict!
    print("Labels Max:", torch.max(target_labels))
    orig_seg_target_id = all_dicts["all"][segmentation_target_object]
    temp = target_labels.eq(orig_seg_target_id).cpu().sum() 
    print("Total points annotated as {} (ID: {}) before converting labels: {}".format(segmentation_target_object, orig_seg_target_id, temp))
    print("Translating labels to the {} dict".format(''.join(args.objects)))
    
    target_labels = dataset.AdaptNumClasses(target_labels, all_dicts).adapt()
    print("Labels Max:", torch.max(target_labels))
    
    # From all the points in the room, find out how many of them belong to 
    # the different objects
    total_points_annotated_as_target_id = target_labels.eq(segmentation_target_object_id).cpu().sum() 
    print("Randomly selected room to visualize: {} (Total points: {})".format(area_and_room, len(data)))
    print("Object to detect: {0} (ID:{1}) (Total annotated {0} points: {2})".format(
        segmentation_target_object, 
        segmentation_target_object_id, 
        total_points_annotated_as_target_id))
    
    if total_points_annotated_as_target_id.item() == 0:
        print("There're no {}s in room {}!".format(segmentation_target_object, area_and_room))
        return
    
    else: 
        # Find out per object info
        # A list of lists:
        # [object, object_ID, 
        # amount of annotated points this object has in this room (as a tensor)
        # amount of predicted points for this object in this room (as a tensor) (initialized to zero)]
        point_breakdown = []
        for k,v in dict_to_use.items():
            point_breakdown.append([k, v, target_labels.eq(v).cpu().sum(), torch.tensor([0])])        
        
        # Work with points_rel (instead of points_abs)
        # Unsquezze the data tensor to give it the depth of batch_size = 1,
        # since we're going to process a single room only
        points = points.unsqueeze(dim = 0)
        
        # Test the model
        # Model input: points.shape([batch_size, room_points, dimensons_per_point)]
        # Model output: preds.shape([batch_size, num_classes, room_points])
        preds, feature_transform, tnet_out, ix = model(points)

        # Output after argmax: preds.shape([batch_size, room_points])
        preds = preds.data.max(1)[1]

        print("Preds Max:", torch.max(preds))
        # Save predictions per object
        for i in point_breakdown:
            # Select the object_id of the element to check accuraracy
            id = i[1]
            # Save predictions for that object
            i[3] = preds.eq(id).cpu().sum()
       
        print(80 * "-")
        print("Model performance (annotated | predicted) points per object:")
        print(80 * "-")
        for obj, id, qty, qty_pred in point_breakdown:
            print("{} (ID:{}): {} | {}".format(obj, id, qty.item(), qty_pred.item()))
        

        #corrects = preds.eq(target_labels.data).cpu().sum()
        #accuracy = corrects.item() / total_points_annotated_as_target_id
        #accuracies.append(accuracy)
        
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
        ones_mask = torch.where(preds == segmentation_target_object_id, 1., 0.).squeeze(dim = 0)
        indices = torch.nonzero(ones_mask).squeeze(dim = 1)
        points = points.squeeze(dim = 0)
        points_to_display = torch.index_select(points, 0, indices)

        # TODO: Insert Lluis' code here for visualization
        # points is the whole room points
        # lluis_code(data, segmentation_target_object_id, points_to_display) 
        
        
        #logger.writer.add_scalar(goal.capitalize() + " Accuracy/Visualization", accuracy)

        #mean_accuracy = (torch.FloatTensor(accuracies).sum()/len(accuracies))*100
        #print(80 * "-")
        #print("Average accuracy: {:.2f}%".format(float(mean_accuracy)))
        #print(80 * "-")


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

    # Get the dicts we'll use to translate:
    #  - from all the objects ID we have in the summary file [0-13]
    #  - to a subset of object IDs: movable [0-5], structural [0-8]
    # when not working with all the num_classes
    all_dicts = summary_file.get_labels()

    # Create the ground truth files for semantic segmentation
    if "segmentation" in args.goal:
        summary_file.label_points_for_semantic_segmentation()
        summary_file.create_sliding_windows()

    # Log insights from the S3DIS dataset into TensorBoard
    logger.log_dataset_stats(summary_file)

    # Logging hparams for future reference
    logger.log_hparams(hparams)
    
    # Define the checkpoint name
    eparams["checkpoint_name"] = "S3DIS_checkpoint_{}_{}_points_{}_dims_{}_num_classes.pth".format(
                                            goal,
                                            hparams["num_points_per_object"] if goal == "classification" else hparams["num_points_per_room"],
                                            hparams["dimensions_per_object"],
                                            hparams["num_classes"],
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
    
    train_ds = getattr(dataset, train_ds_to_call)(eparams['pc_data_path'], all_dicts, transform = None)
    val_ds = getattr(dataset, val_ds_to_call)(eparams['pc_data_path'], all_dicts, transform = None)
    test_ds = getattr(dataset, test_ds_to_call)(eparams['pc_data_path'], all_dicts, transform = None)
    
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
    # (e.g, train_classification(), test_segmentation())
    locals()[task + "_" + goal](model, dataloaders)
    
    if goal == "segmentation":
        # Let's visualize how segmentation works
        #visualize_segmentation_short(model)
        visualize_segmentation_long(model)

    # Close TensorBoard logger and send runs to TensorBoard.dev
    logger.finish()
    #tnet_compare(model, ds)

