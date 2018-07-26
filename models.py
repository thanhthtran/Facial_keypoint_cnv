## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        # Conv lsyer 1-4
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,2)
        self.conv4 = nn.Conv2d(128,256,2)

        #Drop out layer 1-6
        self.drop_out1 = nn.Dropout(p=0.1)
        self.drop_out2 = nn.Dropout(p=0.2)
        self.drop_out3 = nn.Dropout(p=0.3)
        self.drop_out4 = nn.Dropout(p=0.4)
        self.drop_out5 = nn.Dropout(p=0.4)
        self.drop_out6 = nn.Dropout(p=0.4)
        # Dense layer 1-3
        self.fc1 = nn.Linear(12*12*256, 6400)
        self.fc2 = nn.Linear(6400, 1000)
        self.fc3 = nn.Linear(1000, 136)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
   

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        # COnv - Activation-Maxpooling - drop out
        x = self.pool(F.elu(self.conv1(x)))
        x = self.drop_out1(x)
        
        x = self.pool(F.elu(self.conv2(x)))
        x = self.drop_out2(x)
        
        x = self.pool(F.elu(self.conv3(x)))
        x = self.drop_out3(x)
        
        x = self.pool(F.elu(self.conv4(x)))
        x = self.drop_out4(x)
        # Flatten
        x = x.view(x.size(0), -1)
        #Dense
        x = F.elu(self.fc1(x))
        x = self.drop_out5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop_out5(x)
        
        x = self.fc3(x)
    
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x