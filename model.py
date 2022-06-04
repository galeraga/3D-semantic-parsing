from settings import *

class TransformationNet(nn.Module):
    """
    The building block (T-Net) of the PointNet architecture 

    Used to apply both input transform and feature transformations,
    according to Figure 2 of the original PointNet architecture paper

    T-Net aligns all input set to a canonical space before feature extraction.
    How does it do it? It predicts an affine transformation matrix of 3x3 
    to be applied to the coordinate of input points (x, y, z).
    
    T-Net aligns all input set to a canonical space by learning 
    a transformation matrix
    """

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim
        
        # Conv1d for point independent feature extraction
        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(256)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, self.output_dim*self.output_dim)
       

    def forward(self, x):

        # x.shape[batch_size, num_points_per_object, xyzrgb]
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        # Define the Maxpool1D and apply it on x directly
        x = nn.MaxPool1d(num_points, return_indices = False, ceil_mode = False)(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        
        return x


class BasePointNet(nn.Module):
    """
    Defines the PointNet architecture for the classification network,
    according to Figure 2 from the original paper

    It provides the network architecture from the input points to the 
    max pool layer, providing the global feature vector
    """

    def __init__(self, point_dimension):
        super(BasePointNet, self).__init__()
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)
        
        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(1024)
        

    def forward(self, x):

        # x.shape([batch_size, num_points_per_object, dimensions_per_object])
        num_points = x.shape[1]
        
        # Input transformation
        # input_transform.shape([batch_size, dimensions_per_object, dimensions_per_object]) (t-net tensor)
        input_transform = self.input_transform(x) 
        
        # Batch matrix-matrix product 
        # Output: x.shape([[batch_size, num_points_per_object, dimensions_per_object]])
        x = torch.bmm(x, input_transform) 
        
        # After transposing: x.shape([batch_size, dimensions_per_object, num_points_per_object])
        x = x.transpose(2, 1) 
        # tnet_out.shape(c)
        tnet_out=x.cpu().detach().numpy()
        # After relu x.shape(([batch_size, 64, num_points_per_object]))
        x = F.relu(self.bn_1(self.conv_1(x)))
        # After transposing x.shape(([batch_size, num_points_per_object, 64]))
        x = x.transpose(2, 1)
        
        # Feature transformation
        # x.shape([batch_size, num_points_per_object, 64])
        # feature_transform.shape([batch_size, 64, 64])
        feature_transform = self.feature_transform(x) 
        # Output after bmm: x.shape([[batch_size, num_points_per_object, 64]])
        x = torch.bmm(x, feature_transform)
        
        # Saving the features for semantic segmentation 
        segmentation_local_features = x
        
        x = x.transpose(2, 1)
        x = F.relu(self.bn_2(self.conv_2(x)))
        # After relu x.shape[batch_size, 1024, num_points_per_object]
        x = F.relu(self.bn_3(self.conv_3(x))) 
        
        # Max-pooling (x.shape after pooling:[batch_size, 1024, 1])
        x, ix = nn.MaxPool1d(num_points, return_indices = True, ceil_mode = False)(x)  
        
        # Global Feature Vector (shape([batch_size, 1024]))
        global_feature_vector = x.view(-1, 1024)   

        return (global_feature_vector, feature_transform, tnet_out, ix,
                segmentation_local_features)


class ClassificationPointNet(nn.Module):
    """
    Completes the PointNet architecture for the classification network,
    according to Figure 2 from the original paper

    It provides the network architecture from the global feature vector
    to the classification output score
    """

    def __init__(self, num_classes, dropout=0.3, point_dimension = 3):
        
        super(ClassificationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(point_dimension = point_dimension)

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, num_classes)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        """
        x.shape([batch_size, num_points_per_object, dimensions_per_object])
        """
        global_feature_vector, feature_transform, tnet_out, ix_maxpool, seg_local_feats = self.base_pointnet(x)
        
        # global_feature_vector.shape([batch_size, 1024])
        # Relu output: [batch_size, 512]
        x = F.relu(self.bn_1(self.fc_1(global_feature_vector)))
        # Relu output: [batch_size, 256]
        x = F.relu(self.bn_2(self.fc_2(x)))
        # Dropout output: [batch_size, 256]
        x = self.dropout_1(x)
        
        # x.shape([batch_size, num_classes])
        x = self.fc_3(x)

        # preds.shape([batch_size, num_classes])
        preds = F.log_softmax(x, dim=1)
        
        # preds, feature_transform, tnet_out, ix_maxpool
        return preds, feature_transform, tnet_out, ix_maxpool


class SegmentationPointNet(nn.Module):
    """
    Implements the semantic segmentation network of the PointNet architecture

    Concatenates local point features (the feature transformation vector) 
    with global features (the global feature vector) to get the probabality 
    of each point in the cloud

    From https://github.com/yunxiaoshi/pointnet-pytorch/blob/master/pointnet.py
    From https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
    """

    def __init__(self, num_classes, point_dimension = 3):

        super(SegmentationPointNet, self).__init__()
        
        self.num_classes = num_classes
        
        self.base_pointnet = BasePointNet(point_dimension = point_dimension)
        
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.num_classes, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

	
    def forward(self, x):

        num_points = x.shape[1]
        
        # Get the global and local features
        # global_feature_vector.shape([128, 1024])
        # seg_local_feats.shape([batch_size, num_points_per_object, 64])
        global_feature_vector, feature_transform, tnet_out, ix_maxpool, seg_local_feats  = self.base_pointnet(x)

        # Adapt the global feature vector to be concatenated
        global_feature_vector = global_feature_vector.view(-1, 1, 1024).repeat(1, num_points, 1)
        
        # Concatenate global and local features (adding cols)
        # x.shape([batch_size, num_points_per_object, 1088])
        x = torch.cat((seg_local_feats, global_feature_vector), dim = 2)
        
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
    
        x = self.conv4(x)
        
        # Input shape: x.shape(batch_size, num_classes, num_points_per_object)
        # Output shape: x.shape(batch_size, num_points_per_object, num_classes)
        x = x.transpose(2, 1)
        
        # Apply log_softmax over the last dim (num_classes)
        preds = F.log_softmax(x, dim = -1)
        
        # TODO: Preds has to return ([batch_size, num_classes])
        # preds = x.view(batch_size, num_points, self.num_classes)
        
        # Returning the same values than ClassificationPointNet
        # to keep compatatibility in main.py
        return preds, feature_transform, tnet_out, ix_maxpool



