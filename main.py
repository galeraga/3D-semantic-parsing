"""
PointNet implementation with S3DIS dataset
"""

from settings import * 
from dataset import S3DISDataset
from model import ClassificationPointNet, SegmentationPointNet


def test_classification(model, test_dataloader):
    """
    """
    ...
    
def train_classification(model, train_dataloader, val_dataloader):
    """

    """

    # Training
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_loss= np.inf

    for epoch in tqdm(range(hparams['epochs'])):
        epoch_train_loss = []
        epoch_train_acc = []

        # training loop
        for data in train_dataloader:
            points, targets = data  

            """
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()
            if points.shape[0] <= 1:
                continue
            """

            optimizer.zero_grad()
            model = model.train()

            preds, feature_transform, tnet_out, ix_maxpool = model(points)

            # Why?  
            identity = torch.eye(feature_transform.shape[-1])

            if torch.cuda.is_available():
                identity = identity.cuda()
            
            
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
            loss = F.nll_loss(preds, targets) + 0.001 * regularization_loss
            
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
            points, targets = data
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()
            
            model = model.eval()
            preds, feature_transform, tnet_out, ix = model(points)
            loss = F.nll_loss(preds, targets)
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
                os.path.join(
                    eparams['pc_data_path'],
                    'S3DIS_checkpoint_%s.pth' % (hparams['num_points_per_object'])
                    )
                )
            best_loss=np.mean(val_loss)

        train_loss.append(np.mean(epoch_train_loss))
        val_loss.append(np.mean(epoch_val_loss))
        train_acc.append(np.mean(epoch_train_acc))
        val_acc.append(np.mean(epoch_val_acc))


if __name__ == "__main__":

    # Get parser args to decide what the program has to do
    args = parser.parse_args()

    # Adjust some hyperparameters based on the desired load
    if args.load == "low":
        hparams["num_points_per_object"] = 100
        hparams["dimensions_per_object"] = 3
        hparams["epochs"] = 5

    if args.load == "medium":
        hparams["num_points_per_object"] = 1000
        hparams["dimensions_per_object"] = 3
        hparams["epochs"] = 10
        
    if args.load == "high":
        hparams["num_points_per_object"] = 4096
        hparams["dimensions_per_object"] = 6
        hparams["epochs"] = 50
    
    # Create the S3DIS dataset
    ds = S3DISDataset(eparams['pc_data_path'], transform = None)
    print(ds)
    
    # Splitting the dataset (80% training, 10% validation, 10% test)
    # TODO: Modify the dataset to be split by building
    # Building 1 (Area 1, Area 3, Area 6), Building 2 (Area 2, Area 4), Building 3 (Area 5)
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
            shuffle = True
            )
    
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = hparams['batch_size'], 
            shuffle = True
            )
    
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size = hparams['batch_size'], 
            shuffle = False
            )

    # Model instance creation (goal-dependent)
    if args.goal == "classification":
        model = ClassificationPointNet(num_classes = hparams['num_classes'],
                                   point_dimension = hparams['dimensions_per_object'])
    
    if args.goal == "segmentation":
        model = SegmentationPointNet(num_classes = hparams['num_classes'],
                                   point_dimension = hparams['dimensions_per_object'])

    optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

    # Select the task to do
    if args.goal == "classification": 
        if args.task == "train":
            msg = "Starting {}-{} with: ".format(args.task, args.goal)
            msg += "{} points per object | ".format(hparams['num_points_per_object'])
            msg += "{} dimensions per object | ".format(hparams['dimensions_per_object'])
            msg += "{} batch_size ".format(hparams['batch_size'])
            print(msg)
            train_classification(model, train_dataloader, val_dataloader)
        if args.task == "test":
            test_classification(model, test_dataloader)


