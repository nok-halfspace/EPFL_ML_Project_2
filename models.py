import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from constants import * 

# -------------------------------------------
# U-NET
# Architecture based on the initial paper
# -------------------------------------------

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 2):
        super(UNET,self).__init():
            # Contracting path
            self.contract1 = self.contracting_block(in_channels, 64)
            self.contract2 = self.contracting_block(64, 128)
            self.contract3 = self.contracting_block(128, 256)
            self.contract4 = self.contracting_block(256, 512)
                 
            # Expansive path
            self.expand1 = self.expanding_block(512, 1024)
            self.expand2 = self.expanding_block(1024, 512)
            self.expand3 = self.expanding_block(512, 256)
            self.expand4 = self.expanding_block(256, 128)
            
            # Final block
            self.output = self.output_block(128, out_channels)
            
    
    def doubleConv_block(self, in_channels, out_channels):
        """ (conv + ReLU + BN) * 2 times """
        doubleConv_block = torch.nn.Sequential(
             torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3),
            torch.nn.ReLU(),
            # Position of Batch Normalization (BN) wrt nonlinearity unclear, but experiments are generally in favor of this solution, which is the current default(ReLU + BN) 
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels)
        )
        return doubleConv_block
            
    
    def contracting_block(self, in_channels, out_channels):
         """ (conv + ReLU + BN) * 2 times + maxpool """
        contracting_block = torch.nn.Sequential(
            self.doubleConv_block(in_channels, out_channels),
            torch.nn.MaxPool2d(kernel_size = 2)
        )  
        return contracting_block
        

    def expanding_block(self, in_channels, tmp_channels):
        """ (conv + ReLU + BN) * 2 times + upconv """
        out_channels = tmp_channels // 2
        expanding_block = torch.nn.Sequential(
            self.doubleConv_block(in_channels, tmp_channels),
            torch.nn.ConvTranspose2d(in_channels = tmp_channels, out_channels = out_channels, kernel_size = 2, stride = 2)
        )
        return expanding_block
    
    
    def output_block(self, in_channels = 128, out_channels = 2):
        tmp_channels = in_channels // 2
        output_block = torch.nn.Sequential(
            self.doubleConv_block(in_channels, tmp_channels),
            torch.nn.Conv2d(in_channels = tmp_channels, out_channels = out_channels, kernel_size = 1)
        )
        return output_block
     
    def concatenating_block(self, x_contracting, x_expanding):
        delta = (x_contracting.size()[2] - x_expanding.size()[2]) // 2 
        x_cropped = F.pad(x_contracting, (-delta, -delta, -delta, -delta))
        return torch.cat([x_cropped, x_expanding], dim = 1)
     
        
    def forward(self, x): 
        # Padding with reflection 
        # pad size found such that after doing all the convolutions n_out = n_in <=> n_padded = n_in + 184 (to be verified)
        pad = 92
        x = F.pad(x, (pad, pad, pad, pad), mode = 'reflect')
        x1 = self.contract1(x)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contact4(x3)
        # Remove the maxpool from the contracting block to retrieve the output of the layer 
        y1 = self.concatening_block(x
        
        
    
          
            
        
    
 

            
            
            
  
            
            
   def create_UNET():
            network = UNET()
            network.to(DEVICE)
            criterion = 
            optimize = optim.Adam(network.parameters())
            return network, criterion, optimizer
            
        