"""Code Adapted from https://github.com/bzantium/pytorch-admm-pruning/blob/master/model.py"""
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10, criterion=None, device=None):
        super(LeNet, self).__init__()
        self.criterion = criterion
        self.device = device
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, batch, get_prediction=False):
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        if get_prediction:
            return out
        return self.criterion(out, y)