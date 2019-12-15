import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *
import torch.optim as optim



# -------------------------------------------
# U-NET 3rd version
# Architecture based on the initial paper
# -------------------------------------------
class UNET3rd(nn.Module):
    """
    TODO
    """
    def __init__(self):
        super(UNET3rd, self).__init__()

        self.down1 = ContractStep(3, 16)
        self.down2 = ContractStep(16, 32)

        self.down3 = ContractStep(32, 64, p = DROPOUT)
        self.center = ExpandStep(64, 128, 64)
        self.up3 = ExpandStep(128, 64, 32)
        self.up2 = ExpandStep(64, 32, 16)
        self.up1 = LastStep(32, 16, 3) # get the result

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        down1, bridge1 = self.down1(x)
        down2, bridge2 = self.down2(down1)
        down3, bridge3 = self.down3(down2)
        center = self.center(down3)
        up3 = self.up3(torch.cat([center, bridge3], 1))
        up2 = self.up2(torch.cat([up3, bridge2], 1))

        up1 = self.up1(torch.cat([up2, bridge1], 1))

        return self.Sigmoid(up1)

class ContractStep(nn.Module):
    """
    TODO
    """
    def __init__(self, in_channels, out_channels, p = 0):
        super(ContractStep, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.relu(self.conv1(x))
        x = self.batchnorm2(x)
        x = self.relu(self.conv2(x))
        to_concat = x.clone()
        x = self.dropout(x) # Check original

        return self.maxpooling(x), to_concat

class ExpandStep(nn.Module):
    """
    TODO
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ExpandStep, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.batchnorm2 = nn.BatchNorm2d(middle_channels)
        self.upconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.relu(self.conv1(x))

        x = self.batchnorm2(x)
        x = self.relu(self.conv2(x))

        return self.upconv(x)

class LastStep(nn.Module):
    """
    Define a single UNet up step, using convolutions and maxpooling.
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(LastStep, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.batchnorm2 = nn.BatchNorm2d(middle_channels)

        self.final = nn.Conv2d(middle_channels, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.relu(self.conv1(x))
        x = self.batchnorm2(x)
        x = self.relu(self.conv2(x))
        return self.final(x)



def create_UNET3rd():
    network = UNET3rd().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(network.parameters())
    return network, criterion, optimizer
