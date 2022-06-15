"""
PointNet implementation with S3DIS dataset
"""

from settings import * 
from dataset import S3DISDataset4Classification, S3DISDataset4Segmentation 
from model import ClassificationPointNet, SegmentationPointNet
from tensorboardlogger import TensorBoardLogger 
from summarizer import S3DIS_Summarizer


def task_welcome_msg(task = None):
    """
    """
    msg = "Starting {}-{} with: ".format(task, ''.join(args.goal))
    msg += "{} points per object | ".format(hparams['num_points_per_object'])
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
        train(model, dataloaders)
    
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
        preds, feature_transform, tnet_out, ix = model(points)
        
        # preds.shape([batch_size, num_classes])
        preds = preds.data.max(1)[1]
        
        corrects = preds.eq(target_labels.data).cpu().sum()
        accuracy = corrects.item() / float(hparams['batch_size'])
        accuracies.append(accuracy)
        
        logger.writer.add_scalar("Accuracy/Test", accuracy, batch_idx)
    
    mean_accuracy = (torch.FloatTensor(accuracies).sum()/len(accuracies))*100
    print("Average accuracy: {:.2f} ".format(float(mean_accuracy)))
               
def train(model, dataloaders):
    """
    Train the PointNet network

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

    # Aux training vars
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_loss= np.inf

    optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

    # To avoid MaxPool1d warning in GCP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for epoch in tqdm(range(hparams['epochs'])):
            epoch_train_loss = []
            epoch_train_acc = []

            # training loop
            for data in train_dataloader:
                model = model.train()
                
                points, targets = data  
                
                points = points.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                
                preds, feature_transform, tnet_out, ix_maxpool = model(points)

                # Why?  
                identity = torch.eye(feature_transform.shape[-1])

                if torch.cuda.is_available():
                    identity = identity.cuda()
                
                # Formula (2) in original paper (Lreg)
                # TODO: According to the original paper, it should only be applied
                # during the alignment of the feature space (with higher dimension (64))
                # than the spatial transformation matrix (3)
                # With the regularization loss, the model optimization becomes more
                # stable and achieves better performance
                regularization_loss = torch.norm(
                    identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
                

                # Loss: The negative log likelihood loss 
                # It is useful to train a classification problem with C classes.
                # torch.nn.functional.nll_loss(input, target, ...) 
                # input – (N,C) (N: batch_size; C: num_classes) 
                # target - (C)
                # preds.shape[batch_size, num_classes]
                # targets.shape[batch_size], but every item in the target tensor
                # must be in the range of num_classes - 1 
                # E.g: if num_classes = 2 -> target[i] < 2 {0, 1}

                # A regularization loss (with weight 0.001) is added to the softmax
                # classification loss to make the matrix close to ortoghonal
                # (quoted from supplementary info from the original paper)
                
                if "classification" in args.goal: 
                    loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
                
                if "segmentation" in args.goal: 
                    # TODO: loss has to be defined for semantic segmentation
                    # Loss functions for sem seg: https://arxiv.org/abs/2006.14822
                    # Tversky Loss / Focal Tversky Loss seems to be the best...
                    ...

                epoch_train_loss.append(loss.cpu().item())
            
                loss.backward()
                optimizer.step()
                
                preds = preds.data.max(1)[1]
                corrects = preds.eq(targets.data).cpu().sum()
                accuracy = corrects.item() / float(hparams['batch_size'])
                epoch_train_acc.append(accuracy)
                

            epoch_val_loss = []
            epoch_val_acc = []

            # validation loop
            for batch_number, data in enumerate(val_dataloader):
                model = model.eval()
        
                points, targets = data
                points = points.to(device)
                targets = targets.to(device)
                        
                preds, feature_transform, tnet_out, ix = model(points)
                
                if "classification" in args.goal: 
                    loss = F.nll_loss(preds, targets)

                if "segmentation" in args.goal: 
                    # TODO: loss has to be defined for semantic segmentation
                    # Loss functions for sem seg: https://arxiv.org/abs/2006.14822
                    ...
                
                epoch_val_loss.append(loss.cpu().item())
                
                preds = preds.data.max(1)[1]
                corrects = preds.eq(targets.data).cpu().sum()
                accuracy = corrects.item() / float(hparams['batch_size'])
                epoch_val_acc.append(accuracy)


            print('Epoch %s: train loss: %s, val loss: %f, train accuracy: %s,  val accuracy: %f'
                    % (epoch,
                        round(np.mean(epoch_train_loss), 4),
                        round(np.mean(epoch_val_loss), 4),
                        round(np.mean(epoch_train_acc), 4),
                        round(np.mean(epoch_val_acc), 4)))

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
            logger.writer.add_scalar("Loss/Training", train_loss[-1], epoch)
            logger.writer.add_scalar("Loss/Validation", val_loss[-1], epoch)
            logger.writer.add_scalar("Accuracy/Training", train_acc[-1], epoch)
            logger.writer.add_scalar("Accuracy/Validation", val_acc[-1], epoch)


if __name__ == "__main__":

    # Prepare to run on CUDA/CPU
    device = hparams['device']

    # Create a TensorBoard logger instance
    logger = TensorBoardLogger(args)

    # Create the ground truth file
    summary_file = S3DIS_Summarizer(eparams["pc_data_path"], logger)

    # Log insights from the S3DIS dataset into TensorBoard
    logger.log_dataset_stats(summary_file)

    # Logging hparams for future reference
    logger.log_hparams(hparams)
    
    # Define the checkpoint name
    eparams["checkpoint_name"] = "S3DIS_checkpoint_{}_{}_points_{}_dims.pth".format(
                                            ''.join(args.goal),
                                            hparams["num_points_per_object"],
                                            hparams["dimensions_per_object"]
                                            )

    # When choices are given in parser add_argument, the parser returns a list 
    if "classification" in args.goal:
        # Create the S3DIS dataset
        ds = S3DISDataset4Classification(eparams['pc_data_path'], transform = None)
        print(ds)
        
        # Create the dataloaders
        # dataloaders = (train_dataloader, validation_dataloader, test_dataloader)
        # TODO: Redefine the dataloaders based on task (classification, segmentation)
        dataloaders = create_dataloaders(ds)

        # Model instance creation (goal-dependent)
        model = ClassificationPointNet(num_classes = hparams['num_classes'],
                                   point_dimension = hparams['dimensions_per_object']).to(device)
          
        # Select the task to do
        if "train" in args.task:
            train(model, dataloaders)
        
        if "test" in args.task:
            test_classification(model, dataloaders)


    if "segmentation" in args.goal:
        # Create the files for semantic segmentation
        summary_file.label_points_for_semantic_segmentation()
      
        # Create the S3DIS dataset
        ds = S3DISDataset4Segmentation(eparams['pc_data_path'], transform = None)
        print(ds)
        
        # Create the dataloaders
        # dataloaders = (train_dataloader, validation_dataloader, test_dataloader)
        # TODO: Redefine the dataloaders based on task (classification, segmentation)
        dataloaders = create_dataloaders(ds)
        
        # Model instance creation (goal-dependent)
        model = SegmentationPointNet(num_classes = hparams['num_classes'],
                                   point_dimension = hparams['dimensions_per_object']).to(device)
      

        if "train" in args.task:
            train(model, dataloaders)
        
        if "test" in args.task:
            ...
      
    # Close TensorBoard logger and send runs to TensorBoard.dev
    logger.finish()
    
    

