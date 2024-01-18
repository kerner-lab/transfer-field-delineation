"""
Different Skeleton of Unet Models for use in project
"""
from segmentation_models_pytorch import Unet
import torch
from torch import nn
import matplotlib.pyplot as plt


class UNetResNet50_9(nn.Module):
    """A UNet with ResNet50 Encoder with 9 input channels for 3 seasons"""
    def __init__(self, num_classes=1):
        super(UNetResNet50_9, self).__init__()
        self.conv2d = nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.unet_filled = Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        self.unet_border = Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x1 = self.unet_filled(x)
        x2 = self.unet_border(x)
        x1 = self.activation(x1)
        x2 = self.activation(x2)
        return x1, x2
    

class UNetResNet50_3(nn.Module):
    """UNet with ResNet50 Encoder with 3 input channels for 1 season"""
    def __init__(self, num_classes=1):
        super(UNetResNet50_3, self).__init__()
        self.unet_filled = Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        self.unet_border = Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            classes=num_classes,
            activation=None
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.unet_filled(x)
        x1 = self.activation(x1)
        x2 = self.unet_border(x)
        x2 = self.activation(x2)
        return x1, x2
    
if __name__ == '__main__':
    model = UNetResNet50_9()
    x = torch.randn(1, 9, 256, 256)
    y = model(x)
    print(y.shape)
    # plt.imshow(y.squeeze(0).squeeze(0).detach().numpy())
    # plt.show()