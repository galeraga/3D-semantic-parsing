# http://www.open3d.org/docs/latest/introduction.html
# Pay attention to Open3D-Viewer App http://www.open3d.org/docs/latest/introduction.html#open3d-viewer-app
# and the Open3D-ML http://www.open3d.org/docs/latest/introduction.html#open3d-ml


from settings import * 
from summarizer import S3DIS_Summarizer
from dataset import S3DISDataset
from model import ClassificationPointNet

# Define the logging settings
# Logging is Python-version sensitive
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# For Python < 3.9 (minor version: 9) 
# encoding argument can't be used
if sys.version_info[1] < 9:
    logging.basicConfig(filename = os.path.join(eparams['pc_data_path'], eparams['log_file']),
        level=logging.WARNING,
        format='%(asctime)s %(message)s')
else:
    logging.basicConfig(filename = os.path.join(eparams['pc_data_path'], eparams['log_file']),
        encoding='utf-8', 
        level=logging.WARNING,
        format='%(asctime)s %(message)s')


def normalize_RGB_single_file(f):
    """
    Takes the input file and calculates the RGB normalization
    for a single point cloud file
    """

    # Keep the original dataset file intact and create 
    # a new file with normalized RGB values 
    file_path, file_name = os.path.split(f)   
    tgt_file = file_name.split('.')[0] + eparams['pc_file_extension_rgb_norm']
     
    # Skip the process if the file has been already normalized
    if (tgt_file in os.listdir(file_path)) or (eparams['pc_file_extension_rgb_norm'] in file_name):
        print("...skipped (already normalized)")
        return
    else:
        tgt_file = os.path.join(file_path, tgt_file)

    normalized = ''
    with open(f) as src:
        with open(tgt_file, "w") as tgt:
            try:
                for l in src:
                    # Convert the str to list for easier manipulation
                    x, y, z, r, g, b = l.split()
                    r = float(r)/255
                    g = float(g)/255
                    b = float(b)/255

                    # Back to str again
                    normalized += ' '.join([str(x), str(y), str(z), 
                        '{:.8s}'.format(str(r)), 
                        '{:.8s}'.format(str(g)), 
                        '{:.8s}'.format(str(b)), 
                        '\n'])        
                
                tgt.write(normalized)

            except ValueError:
                msg1 = " -> unable to procees file %s " % src.name
                msg2 = msg1 + "(check log at %s)" % os.path.join(eparams['pc_data_path'], eparams['log_file'])
                print(msg2)
                logging.warning(msg1)
            
            else:
                print("...done")


def RGB_normalization(areas):
    """
    Normalize RGB in all disjoint spaces in order to let o3d display them
    """
    # Let's gather the total number of spaces to process    
    total_areas = len(areas)
    total_spaces = 0
    for space in areas:
        total_spaces += len(areas[space])

    # Let's process each space
    total_processed = 0
    for idx, (area, folders) in enumerate(sorted(areas.items())):
        for folder in sorted(folders):    
            total_processed += 1         
            print("Processing RGB normalization in {} ({}/{})| file {} ({}/{})".format(
                area, (idx+1), total_areas, folder, total_processed, total_spaces), 
                end = " ")
            path_to_space = os.path.join(eparams['pc_data_path'], area, folder)
            normalize_RGB_single_file(os.path.join(path_to_space, folder) + eparams['pc_file_extension'])

            # Let's also process the annotations
            path_to_annotations = os.path.join(path_to_space,"Annotations")
            for file in os.listdir(path_to_annotations):
                print("\tProcessing RGB normalization in file: ", file, end = " ")
                normalize_RGB_single_file(os.path.join(path_to_annotations, file))



def get_spaces(path_to_data):
    """
    Inspect the dataset location to determine the amount of available 
    areas and spaces (offices, hallways, etc) 
    Path_to_data\Area_N\office_X
                       \office_Y
                       \office_Z
    Input: Path to dataset
    Output: A dict with 
        - key: Area_N
        - values: a list of included disjoint spaces per Area
    """
    
    # Keep only folders starting with Area_XXX
    areas = dict((folder, '') for folder in os.listdir(path_to_data) if folder.startswith('Area'))
    
    # For every area folder, get the disjoint spaces included within it
    # Removing any file that contains '.' (e.g., .DStore, alignment.txt)
    # os.path.join takes into account the concrete OS separator ("/", "\")
    for area in areas:
        areas[area] = sorted([subfolder for subfolder in os.listdir(os.path.join(path_to_data, area)) 
            if not '.' in subfolder])

    return areas


if __name__ == "__main__":

    # Create the summary file that will contain important info about the dataset
    summary = S3DIS_Summarizer(eparams['pc_data_path'], check_consistency = False)
    
    # Get the labels dict
    # {0: 'openspace', 1: 'pantry', ... , 10: 'lounge'}
    # {0: 'bookcase', 1: 'door', 2: 'ceiling', ... , 13: 'floor'}
    # space_labels_dict, object_labels_dict = summary.get_labels()
    
    # Get statistical info
    # summary.get_stats()

    # Create the S3DIS dataset
    ds = S3DISDataset(eparams['pc_data_path'], transform = None)
    print(ds)
    
    """
    for idx,i in enumerate(ds):
        obj, label = i
        print("{} - Object shape {} | Label: {} ".format(idx, obj.shape, label)) 
    """

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

    """
    num_batches = len(train_dataloader)
    for idx, dl in enumerate(train_dataloader):
        bobject, blabel = dl
        msg = "Checking dataloader {}/{} | "
        msg += "Batch object shape {} | "
        msg += "Batch label lenght {}"
        print(msg.format(idx +1, num_batches, bobject.shape, len(blabel)))
    """ 
    

    # Model instance    
    model = ClassificationPointNet(num_classes = hparams['num_classes'],
                                   point_dimension = hparams['dimensions_per_object'])

    optimizer = optim.Adam(model.parameters(), lr = hparams['learning_rate'])

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

            # Why adding 0.001 and regularization_loss
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


    # TODO: To be removed, since all data is based now in the summary file
    # Get a dict of areas and spaces
    # areas_and_spaces = get_spaces(PC_DATA_PATH)

    # TODO: Rebuild the normalization to be based on the sumamry file,
    # not in the traversal of directories
    # Normalize RGB in all spaces
    # RGB_normalization(areas_and_spaces)

    # To quickly test o3d
    # Two minor issues when working with S3DIS dataset:
    # - Open3D does NOT support TXT file extension, so we have to specify 
    #   the xyzrgb format (check supported file extensions here: 
    #   http://www.open3d.org/docs/latest/tutorial/Basic/file_io.html) 
    # - When working with xyzrgb format, each line contains [x, y, z, r, g, b], 
    #   where r, g, b are in floats of range [0, 1]
    #   So we need to normalize the RGB values from the S3DIS dataset in order 
    #   to allow Open3D to display them

    """
    pcd = o3d.io.read_point_cloud(
        os.path.join(PC_DATA_PATH, TEST_PC + PC_FILE_EXTENSION_RGB_NORM),
        format='xyzrgb')
    print(pcd)
    """
    # o3d.visualization.draw_geometries([pcd])