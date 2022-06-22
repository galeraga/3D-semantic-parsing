"""
PointNet implementation with S3DIS dataset
"""

from settings import * 
import dataset 
import model    
from tensorboardlogger import TensorBoardLogger 
from summarizer import S3DIS_Summarizer
from visualitzation import tnet_compare


def task_welcome_msg(task = None):
    """
    """
    msg = "Starting {}-{} with: ".format(task, goal)
    
    if "classification" in args.goal:
        msg += "{} points per object | ".format(hparams['num_points_per_object'])
    
    if "segmentation" in args.goal:
        msg += "{} points per room | ".format(hparams['max_points_per_space'])

    msg += "{} dimensions per object | ".format(hparams['dimensions_per_object'])
    msg += "{} batch_size | ".format(hparams['batch_size'])
    msg += "workers: {} |".format(hparams['num_workers'])
    
    if hparams['device'] == "cpu":
        msg += " device: {}".format(hparams['device'])
    else:
        msg += " device: {} ({}x {})".format(hparams['device'],
                                        torch.cuda.device_count(),
                                        torch.cuda.get_device_name(0)
                                        )

    print(msg)

def create_dataloaders(ds):
    """
    Creates the dataloaders
    """

    # Splitting the dataset (80% training, 10% validation, 10% test)
    # TODO: Modify the dataset to be split by building
    # Building 1 (Area 1, Area 3, Area 6), Building 2 (Area 2, Area 4), Building 3 (Area 5)
    # Other papers set Area 5 for test
    original_ds_length = len(ds)
    training_ds_length = round(0.8*original_ds_length)
    validation_ds_length = round(0.1*original_ds_length)
    test_ds_length = round(0.1*original_ds_length)

    split_criteria = [training_ds_length, validation_ds_length, test_ds_length]
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.dataset.random_split(ds,
                                            split_criteria,
                                            generator=torch.Generator().manual_seed(1))

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


@torch.no_grad()
def test_segmentation(model, dataloaders):
    """
    Test the PointNet segmentation network
    """
    ...

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
        accuracy = corrects.item() / float(hparams['batch_size'])
        accuracies.append(accuracy)
        
        logger.writer.add_scalar(goal.capitalize() + " Accuracy/Test", accuracy, batch_idx)
    
    mean_accuracy = (torch.FloatTensor(accuracies).sum()/len(accuracies))*100
    print("Average accuracy: {:.2f} ".format(float(mean_accuracy)))
               
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

    # To avoid MaxPool1d warning in GCP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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
                loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
                
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
                
                loss = F.nll_loss(preds, targets)
                
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

    # To avoid MaxPool1d warning in GCP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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

    # Create the ground truth files for semantic segmentation
    if "segmentation" in args.goal:
        summary_file.label_points_for_semantic_segmentation()

    # Log insights from the S3DIS dataset into TensorBoard
    logger.log_dataset_stats(summary_file)

    # Logging hparams for future reference
    logger.log_hparams(hparams)
    
    # Define the checkpoint name
    eparams["checkpoint_name"] = "S3DIS_checkpoint_{}_{}_points_{}_dims.pth".format(
                                            goal,
                                            hparams["num_points_per_object"],
                                            hparams["dimensions_per_object"]
                                            )
    
    # Dataset instance creation (goal-dependent) 
    # If goal == classification -> S3DISDataset4Classification
    # If goal == segmentation -> S3DISDataset4Segmentation
    ds_to_call = "S3DISDataset4" + goal.capitalize()  
    ds = getattr(dataset, ds_to_call)(eparams['pc_data_path'], transform = None)
    print(ds)
    
    # Create the dataloaders
    # dataloaders = (train_dataloader, validation_dataloader, test_dataloader)
    dataloaders = create_dataloaders(ds)

    # Model instance creation (goal-dependent)
    # If goal == classification -> ClassificationPointNet
    # If goal == segmentation -> SegmentationPointNet
    model_to_call = goal.capitalize() + "PointNet"
    model = getattr(model, model_to_call)(num_classes = hparams['num_classes'],
                                   point_dimension = hparams['dimensions_per_object']).to(device)


    # Carry out the the task to do
    # (e.g, train_classification(), test_segmentation())
    locals()[task + "_" + goal](model, dataloaders)
    
    # Close TensorBoard logger and send runs to TensorBoard.dev
    logger.finish()
    tnet_compare(model, ds)
