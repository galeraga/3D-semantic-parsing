"""
PointNet implementation with S3DIS dataset
"""

#------------------------------------------------------------------------------
# IMPORTS
#------------------------------------------------------------------------------
from settings import * 
import dataset 
import model    
from tensorboardlogger import TensorBoardLogger 
from summarizer import S3DIS_Summarizer
import visualization 

# from visualization import render_segmentation
#from visualitzation import tnet_compare, tnet_compare_infer,  infer

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

    print(80 * "-")
    print(task.upper())
    print(80 * "-")

    msg = "Starting {}-{} with:".format(task, goal)
    msg += "\n- {} classes ({})".format(hparams['num_classes'], objects_dict)
    
    if "classification" in args.goal:
        msg += "\n- {} points per object ".format(hparams['num_points_per_object'])
    
    if "segmentation" in args.goal:
        msg += "\n- {} points per sliding window ".format(hparams['num_points_per_room'])

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

def load_checkpoint(model):
    """
    """
     # Path to the checkpoint file
    model_checkpoint = os.path.join(
            eparams['pc_data_path'], 
            eparams['checkpoints_folder'], 
            eparams["checkpoint_name"]
            )
    
    # If the checkpoint does not exist, train the model
    if not os.path.exists(model_checkpoint):
        print("\n -> The model does not seem already trained! Starting the training rigth now from scratch...\n")
        task = "train"
        run_model(model, dataloaders, task)
    
    # Loading the existing checkpoint
    print("Loading checkpoint {} ...".format(model_checkpoint))
    state = torch.load(
                model_checkpoint, 
                map_location = torch.device(hparams["device"]))
    model.load_state_dict(state['model'])  
    
    return model

def compute_confusion_matrix(y_true, y_preds):
    """
    Calculate and print confusion matrix and other several metrics 
    to evaluate model performance.

    Args:
    - y_true: A vector containing ground truth labels
    - y_preds: A vector containing the model output prediction
    
    Annotation and predictions must be entered as text to work
    
    Returns:
        - F1 Score (Macro), 
        - F1 Score (Micro),
        - F1 Score (Weighted),
        - Intersection over Union (IoU) (Macro)
        - Intersection over Union (IoU) (Micro)
        - Intersection over Union (IoU) (Weighted) 
        
    More info about thsese metrics can be found here:
    https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix

    """
    # Replace numbers per text
    # Create a revser dict to speed up process of replacing numbers
    reverse_objects_dict = dict()
    for k,v in objects_dict.items():
        reverse_objects_dict[v] = k

    y_preds_text = []
    y_true_text = []
    for n in y_preds:
        y_preds_text.append(reverse_objects_dict[n])
    
    for n in y_true:
        # Preds are ints, targets are floats
        y_true_text.append(reverse_objects_dict[int(n)])

    # Compute confusion matrix
    cm = confusion_matrix(y_true_text, y_preds_text, labels = [k for k in objects_dict])
    
    # Get other metrics
    precision, recall, fscore, support = precision_recall_fscore_support(y_true_text, y_preds_text, labels = [k for k in objects_dict])
    
    # Print the table
    cm_table = PrettyTable()
    per_object_scores = PrettyTable()
    avg_scores = PrettyTable()
    
    cm_table.field_names = ["Object"] + [k for k in objects_dict]
    for idx, row in enumerate(cm.tolist()):
        row = [reverse_objects_dict[idx]] + row
        cm_table.add_row(row) 
    print("\nConfusion Matrix")
    print(cm_table)
    print("")
    
    # Per object Precision, Recall and F1 scores
    print("\nScores (per object)")
    per_object_scores.field_names = ["Scores"] + [k for k in objects_dict]
    per_object_scores.add_row(["Precision"] + ["{:.4f}".format(v) for v in precision.tolist()])
    per_object_scores.add_row(["Recall"] + ["{:.4f}".format(v) for v in recall.tolist()])
    per_object_scores.add_row(["F1 Score"] + ["{:.4f}".format(v) for v in fscore.tolist()])
    print(per_object_scores)
    
    # Average scores
    # 1.- F1
    print("\nScores (averages)")
    f1_score_macro = f1_score(y_true_text, y_preds_text, average = 'macro')
    f1_score_micro = f1_score(y_true_text, y_preds_text, average = 'micro')
    f1_score_weighted = f1_score(y_true_text, y_preds_text, average = 'weighted')

    # 2.-Intersection over union
    iou_score_macro = jaccard_score(y_true_text, y_preds_text, average = "macro")
    iou_score_micro = jaccard_score(y_true_text, y_preds_text, average = "micro")
    iou_score_weighted = jaccard_score(y_true_text, y_preds_text, average = "weighted")
   
    avg_scores.field_names = ["Score", "Macro", "Micro", "Weighted"]  
    avg_scores.add_row(["F1", "{:.4f}".format(f1_score_macro), "{:.4f}".format(f1_score_micro), "{:.4f}".format(f1_score_weighted)])
    avg_scores.add_row(["IoU", "{:.4f}".format(iou_score_macro), "{:.4f}".format(iou_score_micro), "{:.4f}".format(iou_score_weighted)])
    print(avg_scores)
    print("")
    

    # From: https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    #print("Overall accuracy: {}\n".format(ACC))

    return (f1_score_macro, f1_score_micro, f1_score_weighted, 
            iou_score_macro, iou_score_micro, iou_score_weighted)
    

#------------------------------------------------------------------------------
# CLASSIFICATION AND SEGMENTATION METHODS FOR TRAINING, VALIDATION AND TESTING
#------------------------------------------------------------------------------

def process_single_epoch(model, dataloader, optimizer, epoch, task):
    """
    """

    # Aux vars for grand totals
    epoch_loss = []
    epoch_acc = []
    epoch_y_true = []
    epoch_y_preds = []

    tqdm_desc = "{}ing epoch ({}/{})".format(task.capitalize(), epoch, hparams['epochs'])
    for data in tqdm(dataloader, desc = tqdm_desc):
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
        epoch_loss.append(loss.cpu().item())
        
        #loss.requires_grad=True #might need toggle on for watch 
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
        epoch_acc.append(accuracy)

        # Prepare data for confusion matrix    
        targets = targets.view(-1, targets.numel()).squeeze(dim = 0).tolist()
        preds = preds.view(-1, preds.numel()).squeeze(dim = 0).tolist()
        epoch_y_true.extend(targets)
        epoch_y_preds.extend(preds)

    print("\nLoss: ", round(np.mean(epoch_loss), 4))    
    print("Accuracy:", round(np.mean(epoch_acc), 4))    

    f1_scores = compute_confusion_matrix(epoch_y_true, epoch_y_preds)
    
    # For classification only, display the tnet output
    if goal == "classification":
        print(targets)
        visualization.tnet_compare(points, targets, preds, tnet_out, objects_dict, logger)
    
    return (epoch_y_true, epoch_y_preds, epoch_loss, epoch_acc, f1_scores)


@avoid_MaxPool1d_warning
def run_model(model, dataloaders, task):
    """
    """
     # Task welcome message
    task_welcome_msg(task)

    # Get the proper dataloader for the proper task
    if task == "train":
        dataloader = dataloaders[0]
        model = model.train()
    elif task == "validation":
        dataloader = dataloaders[1]
        model = load_checkpoint(model)
        model = model.eval()
    elif task == "test":
        dataloader = dataloaders[2]
        model = load_checkpoint(model)
        model = model.eval()
    elif task == "watch":
        dataloader = dataloaders[2]
        model = load_checkpoint(model)
        model = model.eval()
        hparams["epochs"] = 1
    
    
    # Aux vars to store labels for predictions (to be used by the confusion matrix)
    total_y_true = []
    total_y_preds = []
    total_loss = []
    total_acc = []
    time_per_epoch = []
    # Set the initial best loss to infinity
    best_loss= np.inf

    optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

    for epoch in range(1, hparams['epochs'] +1):
        
        # Time it
        epoch_start_time = datetime.datetime.now()

        # Run the model
        scores = process_single_epoch(model, dataloader, optimizer, epoch, task)
        
        # Save time (in secs)
        epoch_end_time = datetime.datetime.now()
        time_per_epoch.append((epoch_end_time - epoch_start_time).seconds)
        
        # Split results
        total_y_true.extend(scores[0])
        total_y_preds.extend(scores[1])
        total_loss.extend(scores[2])
        total_acc.extend(scores[3])
        f1_score_macro, f1_score_micro, f1_score_weighted, iou_macro, iou_micro, iou_weighted = scores[4]

        # Save the model only when training
        if (np.mean(total_loss) < best_loss) and (task == "train"):
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
            best_loss=np.mean(total_loss)

        # Log results to TensorBoard for every epoch
        base_msg = goal.capitalize() + "/" + task.capitalize()   
        logger.writer.add_scalar(base_msg + " Loss", total_loss[-1], epoch)
        logger.writer.add_scalar(base_msg + " Accuracy", total_acc[-1], epoch)
        logger.writer.add_scalar(base_msg + " Time (secs)", time_per_epoch[-1], epoch)
        logger.writer.add_scalar(base_msg + " F1 Score (Macro)", f1_score_macro, epoch)
        logger.writer.add_scalar(base_msg + " F1 Score (Micro)", f1_score_micro, epoch)
        logger.writer.add_scalar(base_msg + " F1 Score (Weighted)", f1_score_weighted, epoch)
        logger.writer.add_scalar(base_msg + " IoU Score (Macro)", iou_macro, epoch)
        logger.writer.add_scalar(base_msg + " IoU Score (Micro)", iou_micro, epoch)
        logger.writer.add_scalar(base_msg + " IoU Score (Weighted)", iou_weighted, epoch)
        
    # Print confusion matrix in console
    print(80 * "-")
    print("Overall Confusion Matrix")
    print("Task: {}".format(task.capitalize()))
    print("Checkpoint: {}".format(eparams["checkpoint_name"]))
    print(80 * "-")
    compute_confusion_matrix(total_y_true, total_y_preds)

    # Log confusion matrix in TensorBoard
    cf_matrix = confusion_matrix(total_y_true, total_y_preds)    

    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in objects_dict],
                        columns = [i for i in objects_dict])


    points = hparams["num_points_per_room"] if goal == "segmentation" else hparams["num_points_per_object"]
    msg = "Confusion Matrix" + " " + goal.capitalize() + "/"
    msg += task.capitalize() + " " + str(points) + " points" + " " 
    msg += str(hparams["epochs"]) + " epochs"  + " " + chosen_params + " "
    msg += str(hparams["dimensions_per_object"]) + " dimensions per object"
    
    #logger.writer.add_figure(msg, sns.heatmap(df_cm, annot=True).get_figure())
    logger.writer.add_figure(msg, sns.heatmap(df_cm, annot=True).get_figure())

    
#------------------------------------------------------------------------------
# SEMANTIC SEGMENTATION VISUALIZATION
#------------------------------------------------------------------------------

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

    At least, two ways can be followed to achieve this goal:
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

    # Define variables to compute confusion matrices
    per_win_y_true = []
    per_win_y_preds = []
    total_y_true = []
    total_y_preds = []

    # Load checkpoint
    model = load_checkpoint(model)
  
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
        points_abs = data[:, 6:9].to(device)
        target_labels = data[:, -1].to(device)

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

         # Prepare data for confusion matrix    
        targets = target_labels.view(-1, target_labels.numel()).squeeze(dim = 0).tolist()
        preds = preds.view(-1, preds.numel()).squeeze(dim = 0).tolist()
        per_win_y_true.append(targets)
        per_win_y_preds.append(preds) 
        
    # Computing scores for sliding windows
    print("Per sliding window")             
    for i in range(len(all_sliding_windows_for_a_room)):
        print("WinID: {} ".format(i) + 60 * "-")
        compute_confusion_matrix(per_win_y_true[i], per_win_y_preds[i])
    
    print(60 * "-")
    print("Grand totals per room {}:".format(area_and_office))
    print(60 * "-")
    # Create a single list from all the lists
    for i in per_win_y_true:
        total_y_true.extend(i)

    for i in per_win_y_preds:
        total_y_preds.extend(i)

    compute_confusion_matrix(total_y_true, total_y_preds)

    # Visualize ground truth and resultant segmented points
    visualization.render_segmentation(dict_to_use = dict_to_use,
                        str_area_and_office = area_and_office,
                        dict_model_segmented_points = out_dict,
                        b_multiple_seg = True,    
                        b_hide_wall = True,                                  
                        draw_original_rgb_data = False,
                        b_show_room_points = False)   

#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------
if __name__ == "__main__":

    # When choices are given in parser add_argument, 
    # the parser returns a list 
    # goal -> either "classification" or "segmentation"
    # task -> either "train", "validation" or "test"
    goal = ''.join(args.goal)
    task = ''.join(args.task)
    
    # Prepare to run on CUDA/CPU
    device = hparams['device']

    # Create a TensorBoard logger instance
    logger = TensorBoardLogger(args)

    # Create the ground truth file for classification
    summary_file = S3DIS_Summarizer(eparams["pc_data_path"], logger)

    # Get the dicts
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
    eparams["checkpoint_name"] = "S3DIS_checkpoint_{}_{}_points_{}_dims_{}_num_classes_{}_epochs_{}.pth".format(
                                            goal,
                                            hparams["num_points_per_object"] if goal == "classification" else hparams["num_points_per_room"],
                                            hparams["dimensions_per_object"],
                                            hparams["num_classes"],
                                            hparams["epochs"],
                                            chosen_params
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
    if task != "watch":
        run_model(model, dataloaders, task)
    else:
        watch_segmentation(model, dataloaders)
    
    # tnet_compare example here -----------------------
    # Extracting tnet_out and preds:
    #sample = (ds[0])[0]
    #preds,tnet_out = infer(model, sample[0])
    #logger.writer.add_figure('Tnet-out-fig.png', tnet_compare(sample[0], preds, tnet_out), global_step=None, close=True, walltime=None)
    # Using the _infer version that extracts the variables by itself:
    #logger.writer.add_figure('Tnet-out-fig.png', tnet_compare_infer(model, sample[0]), global_step=None, close=True, walltime=None)
    # ---------------------------------------------------

    # We need to close the writer and the logger:
    # Close TensorBoard logger and send runs to TensorBoard.dev
    logger.writer.flush()
    logger.writer.close()
    logger.finish()





