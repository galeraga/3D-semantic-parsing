
from settings import *

class TensorBoardLogger():

    def __init__(self, args, model):
        # Define the folder where we will store all the tensorboard logs
        logdir = os.path.join(eparams['pc_data_path'], 
            eparams['tensorboard_log_dir'],
            f"{args.goal}-{args.task}-{args.load}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # Initialize Tensorboard Writer with the previous folder 'logdir'
        self.writer = SummaryWriter(logdir)
        

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

