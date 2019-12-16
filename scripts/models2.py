import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *
import torch.optim as optim



# -------------------------------------------
# U-NET
# Our architecture
# -------------------------------------------
class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()
        
        # Maxpooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout
        self.dropout = nn.modules.Dropout(0.20)
        
        # Contracting path : Context
        self.contract1 = self.doubleConv_block(3, 16, 16)
        self.contract2 = self.doubleConv_block(16, 32, 32)
        self.contract3 = self.doubleConv_block(32, 64, 64)
        
        # Center of the network
        self.center = self.expanding_block(64, 128, 64)

       # Expanding path : Localization       
        self.expand3 = self.expanding_block(128, 64, 32)
        self.expand2 = self.expanding_block(64, 32, 16)
        self.expand1 = self.output_block(32, 16)

        # Getting the prediction
        self.Sigmoid = nn.Sigmoid()
        
    def doubleConv_block(self, in_channels, tmp_channels, out_channels):
        """ (BN + conv + ReLU) * 2 times """
        doubleConv_block = nn.Sequential(nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, tmp_channels, kernel_size =3, padding = 1),
        nn.ReLU(),  
        nn.BatchNorm2d(tmp_channels),
        nn.Conv2d(tmp_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU()                                
        )

        return doubleConv_block                
        
    def concatenating_block(self, x_contracting, x_expanding):
        return torch.cat([x_expanding, x_contracting], dim = 1)
    
    def expanding_block(self, in_channels, tmp_channels, out_channels):
        """ (BN + conv + ReLU) * 2 times + upconv """
        expanding_block = nn.Sequential(
                 self.doubleConv_block(in_channels, tmp_channels, tmp_channels),
                 nn.ConvTranspose2d(tmp_channels, out_channels, kernel_size = 2, stride = 2)
        )
        return expanding_block    
    
   
    
    def output_block(self, in_channels, tmp_channels):
        output_block = nn.Sequential(
                self.doubleConv_block(in_channels, tmp_channels, tmp_channels),
                nn.Conv2d(tmp_channels, 1, kernel_size=1)
        )
        return output_block
            
    
    def forward(self, layer0):
        layer1_descending = self.contract1(layer0)
        layer2_descending = self.contract2(self.maxpool(layer1_descending))
        layer3_descending = self.contract3(self.maxpool(layer2_descending))
        layer3_descending = self.dropout(layer3_descending)
        layer_center = self.center(self.maxpool(layer3_descending))

        layer3_ascending = self.expand3(self.concatenating_block(layer3_descending, layer_center))
        layer2_ascending = self.expand2(self.concatenating_block(layer2_descending, layer3_ascending))
        layer1_ascending = self.expand1(self.concatenating_block(layer1_descending, layer2_ascending))
        
        output = self.Sigmoid(layer1_ascending)

        return output


def create_UNET():
    network = UNET().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(network.parameters())
    return network, criterion, optimizer