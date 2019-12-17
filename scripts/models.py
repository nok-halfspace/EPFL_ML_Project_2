import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *
import torch.optim as optim


# -------------------------------------------
# U-NET
# Our architecture
# -------------------------------------------
class smaller_UNET(nn.Module):
    def __init__(self):
        super(smaller_UNET,self).__init__()
        
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


def create_smaller_UNET():
    network = smaller_UNET().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(network.parameters())
    return network, criterion, optimizer



# -------------------------------------------
# U-NET
# Initial architecture from the paper 
# -------------------------------------------
class UNET(nn.Module):
    def __init__(self):
        super(UNET,self).__init__()

        # Contracting path
        self.contract1 = self.doubleConv_block(3, 64, 0.25)
        self.contract2 = self.doubleConv_block(64, 128)
        self.contract3 = self.doubleConv_block(128, 256)
        self.contract4 = self.doubleConv_block(256, 512)

        self.maxpool = torch.nn.MaxPool2d(kernel_size = 2)

        # Expansive path
        self.expand5 = self.expanding_block(512, 1024)
        self.expand4 = self.expanding_block(1024, 512)
        self.expand3 = self.expanding_block(512, 256)
        self.expand2 = self.expanding_block(256, 128)

        # Final block
        self.output = self.output_block(128, 1)
        
        # Getting the prediction
        self.Sigmoid = nn.Sigmoid()


    def doubleConv_block(self, in_channels, out_channels, p_dropout = 0.5):
        """ (conv + ReLU + BN) * 2 times """
        doubleConv_block = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3),
        torch.nn.ReLU(), 
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.modules.Dropout(p_dropout)
        )

        return doubleConv_block



    def expanding_block(self, in_channels, tmp_channels):
        """ (conv + ReLU + BN) * 2 times + upconv """
        out_channels = tmp_channels // 2
        expanding_block = torch.nn.Sequential(
            self.doubleConv_block(in_channels, tmp_channels),
            torch.nn.ConvTranspose2d(tmp_channels, out_channels, kernel_size = 2, stride = 2)
        )
        return expanding_block


    def output_block(self, in_channels = 128, out_channels = 1):
        tmp_channels = in_channels // 2
        output_block = torch.nn.Sequential(
            self.doubleConv_block(in_channels, tmp_channels),
            torch.nn.Conv2d(in_channels = tmp_channels, out_channels = out_channels, kernel_size = 1),
        )
        return output_block

    def concatenating_block(self, x_contracting, x_expanding):
        delta2 = (x_contracting.size()[2] - x_expanding.size()[2])
        delta = delta2 // 2
        if delta2 % 2 == 0 :
        # see which type of padding to apply
            x_cropped = F.pad(x_contracting, (-delta, -delta, -delta, -delta))
        else :
            x_cropped = F.pad(x_contracting, (-delta - 1, -delta, -delta - 1, -delta))
        return torch.cat([x_cropped, x_expanding], dim = 1)


    def forward(self, layer0):
        # Padding with reflection
        # pad size found such that after doing all the convolutions n_out = n_in <=> n_padded = n_in + 194 (to be verified)
        # Can't obtain a perfect padding to obtain a 200 * 200 image in the end : pad of 93 => final image 388 * 388 / pad of 94 => 404 * 404
         # Drop-out layers based on https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
        pad = 94
        layer0 = F.pad(layer0, (pad, pad, pad, pad), mode = 'reflect')
        layer1_descending = self.contract1(layer0)
        layer2_descending = self.contract2(self.maxpool(layer1_descending))
        layer3_descending = self.contract3(self.maxpool(layer2_descending))
        layer4_descending = self.contract4(self.maxpool(layer3_descending))
        layer5 = self.maxpool(layer4_descending)

        # _ascending = input of the layer
        layer4_ascending = self.expand5(layer5)
        layer3_ascending = self.expand4(self.concatenating_block(layer4_descending, layer4_ascending))
        layer2_ascending = self.expand3(self.concatenating_block(layer3_descending, layer3_ascending))
        layer1_ascending = self.expand2(self.concatenating_block(layer2_descending, layer2_ascending))
        output = self.output(self.concatenating_block(layer1_descending, layer1_ascending))
        
        output = self.Sigmoid(output)


        return output


def create_UNET():
    network = UNET().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = optim.Adam(network.parameters())
    return network, criterion, optimizer






