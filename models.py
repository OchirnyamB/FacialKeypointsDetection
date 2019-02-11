## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1))

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2,2)
 
        # Dropout Layers 
        # They essentially turn off certain nodes in a layer with some probability p.
        # This ensures that all nodes get an equal chance to try and classify different images during training.
        # and it reduces the likelihood that only a few, heavily-weighted nodes will dominate the process.
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        
        # Fully connected layers
        self.fcl1 = nn.Linear(43264, 10000)
        self.fcl2 = nn.Linear(10000, 1000)
        self.fcl3 = nn.Linear(1000, 136)
  
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # Convulation -> Max pooling -> Dropout Layers
        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the image
        x = x.view(x.size(0), -1) 

        # Fully connected layers -> Dropout Layers
        x = self.drop2(F.elu(self.fcl1(x)))
        x = self.drop3(F.elu(self.fcl2(x)))
        x = self.drop4(F.elu(self.fcl3(x)))

        # a modified x, having gone through all the layers of your model, should be returned
        return x
