import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, resnet50

class ModifiedResNet50(nn.Module):

    def __init__(self):

        super(ModifiedResNet50, self).__init__()

        # Load a pre-trained ResNet-50 model
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the fully connected layer and the adaptive pooling layer
        self.resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Convolutional layer to transform the feature map from 2048 to 768 depth
        self.conv_transform = nn.Conv2d(2048, 768, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Pass input through ResNet feature extractor
        features = self.resnet_feature_extractor(x)
        # Transform features from 2048 channels to 768 channels
        transformed_features = self.conv_transform(features)

        return transformed_features
