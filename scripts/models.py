import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *
import torch.optim as optim



# -------------------------------------------
# U-NET
# Architecture based on the initial paper
# -------------------------------------------
class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 2):
        super(UNET,self).__init__()
        # Contracting path
        self.contract1 = self.doubleConv_block(in_channels, 64)
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
        self.output = self.output_block(128, out_channels)



    def doubleConv_block(self, in_channels, out_channels):
        """ (conv + ReLU + BN) * 2 times """
        doubleConv_block = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3),
        torch.nn.ReLU(),  # apply activqtion function
        # Position of Batch Normalization (BN) wrt nonlinearity unclear, but experiments are generally in favor of this solution, which is the current default(ReLU + BN)
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels))

        return doubleConv_block



    def expanding_block(self, in_channels, tmp_channels):
        """ (conv + ReLU + BN) * 2 times + upconv """
        out_channels = tmp_channels // 2
        expanding_block = torch.nn.Sequential(
            self.doubleConv_block(in_channels, tmp_channels),
            torch.nn.ConvTranspose2d(tmp_channels, out_channels, kernel_size = 2, stride = 2)
        )
        return expanding_block


    def output_block(self, in_channels = 128, out_channels = 2):
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
        pad = 94
        layer0 = F.pad(layer0, (pad, pad, pad, pad), mode = 'reflect')
        print('layer0', layer0.shape)

        layer = self.contract1(layer0)
        print('layer1d', layer.shape)
        layer2_descending = self.contract2(self.maxpool(layer1_descending))
        print('layer2d', layer2_descending.shape)
        layer3_descending = self.contract3(self.maxpool(layer2_descending))
        print('layer3d', layer3_descending.shape)
        layer4_descending = self.contract4(self.maxpool(layer3_descending))
        print('layer4d', layer4_descending.shape)
        layer5 = self.maxpool(layer4_descending)
        print('layer5', layer5.shape)


        # _ascending = input of the layer
        layer4_ascending = self.expand5(layer5)
        print('layer4a', layer4_ascending.shape)
        layer3_ascending = self.expand4(self.concatenating_block(layer4_descending, layer4_ascending))
        print('layer3a', layer3_ascending.shape)
        layer2_ascending = self.expand3(self.concatenating_block(layer3_descending, layer3_ascending))
        print('layer2a', layer2_ascending.shape)
        layer1_ascending = self.expand2(self.concatenating_block(layer2_descending, layer2_ascending))
        print('layer1a', layer1_ascending.shape)

        output = self.output(self.concatenating_block(layer1_descending, layer1_ascending))
        print('output', output.shape)

        # Converting into vector
        #output = output.view(TRAINING_SIZE, N_CLASSES, -1)
        #print('output vector', output.shape)
        #finally no need to convert into vector with nn.crossentropy.loss

        return output


def create_UNET():
    network = UNET()
    network.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters())
    return network, criterion, optimizer
