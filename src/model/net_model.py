import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, number_class):
        super().__init__() 
        self.number_class = number_class
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 9, 128)
        self.conv3 = nn.Conv2d(9, 12, 256)
        self.fc1 = nn.Linear(9*12*256, 120)
        self.fc2 = nn.Linear(60, 30)
        self.fc2 = nn.Linear(80, number_class)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x














        

       







