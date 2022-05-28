
from settings import *

class TensorBoardLogger():

    def __init__(self, args):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join(eparams['pc_data_path'], 
            eparams['tensorboard_log_dir'],
            f"{args.goal}-{args.task}-{args.load}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer = SummaryWriter(logdir)
        
    def log_dataset_stats(self, summary_file):
        """
        Get insights from the dataset info, based on the ground truth
        (aka summary) file
        """
        # Open the CSV summary file
        path_to_summary_file = os.path.join(eparams['pc_data_path'], eparams['s3dis_summary_file'])

        # Get the whole summary
        summary = pd.read_csv(path_to_summary_file, 
            header =0, 
            usecols = summary_file.S3DIS_summary_cols, 
            sep = "\t"
            )
        
        # Info per area
        areas = sorted(set(summary['Area']))

        for idx, area in enumerate(areas):     
            # Returns a new dataframe containing only the proper area
            area_df = summary.loc[summary['Area'] == area]

            # Spaces per area
            # For that area, get non-repeated spaces
            self.writer.add_scalar("S3DIS Dataset/Spaces per area", 
                len(sorted(set(area_df["Space"]))), 
                idx + 1
                )
            
            # Points per area
            self.writer.add_scalar("S3DIS Dataset/Points per area", 
                area_df["Object Points"].sum(), 
                idx + 1
                )

        # Info per spaces
        spaces = sorted(set(summary['Space']))

        for idx, space in enumerate(spaces):
            # Returns a new dataframe containing only the spaces
            space_df = summary.loc[summary['Space'] == space]

            objects_per_area = len(sorted(set(space_df["Object Points"])))
            print("Space: {}, number of objects: {}".format(space, objects_per_area))
        
        # TODO: Points per area
        for area in areas:
            area_df = summary.loc[summary['Area'] == area]
            area_points = area_df["Object Points"]
            print("Points per {}: {}".format(area, area_points.sum()))

        # TODO: Points per space:
        for space in spaces:
            space_df = summary.loc[summary['Space'] == space]
            space_points = space_df["Object Points"]
            #print("Points per {}: {}".format(space, space_points.sum()))
        
            # Points per kind of space.
            # He comprovat que els valors que aquests valors son les sumes dels anteriors.
            summary_spaces = summary.groupby(summary["Space"].str.split('_').str[0]).sum()
            print(summary_spaces)



    def log_reconstruction_training(self, model, epoch, train_loss_avg, val_loss_avg, reconstruction_grid):

        # TODO: Log train reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/train_loss" as tag


        # TODO: Log validation reconstruction loss to tensorboard.
        #  Tip: use "Reconstruction/val_loss" as tag


        # TODO: Log a batch of reconstructed images from the validation set.
        #  Use the reconstruction_grid variable returned above.


        # TODO: Log the weights values and grads histograms.
        #  Tip: use f"{name}/value" and f"{name}/grad" as tags
        for name, weight in model.encoder.named_parameters():
            continue # remove this line when you complete the code


        pass


    def log_classification_training(self, model, epoch, train_loss_avg,
                                    val_loss_avg, val_acc_avg, train_acc_avg,
                                    fig):
        # TODO: Log confusion matrix figure to tensorboard

        # TODO: Log validation loss to tensorboard.
        #  Tip: use "Classification/val_loss" as tag


        # TODO: Log validation accuracy to tensorboard.
        #  Tip: use "Classification/val_acc" as tag


        # TODO: Log training loss to tensorboard.
        #  Tip: use "Classification/train_loss" as tag


        # TODO: Log training accuracy to tensorboard.
        #  Tip: use "Classification/train_acc" as tag


        pass


    def log_model_graph(self, model, dataloader):
        
        batch, _ = next(iter(dataloader))
        """
        TODO:
        We are going to log the graph of the model to Tensorboard. For that, we need to
        provide an instance of the model and a batch of images, like you'd
        do in a forward pass.
        """


    def log_embeddings(self, model, train_loader, device):
        list_latent = []
        list_images = []
        for i in range(10):
            batch, _ = next(iter(train_loader))

            # forward batch through the encoder
            list_latent.append(model.encoder(batch.to(device)))
            list_images.append(batch)

        latent = torch.cat(list_latent)
        images = torch.cat(list_images)

        # TODO: Log latent representations (embeddings) with their corresponding labels (images)


        # Be patient! Projector logs can take a while

