import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from skimage import img_as_ubyte
from prepareData import *
from constants import *

MAX_VALUE = 256

""" Converts unit pixels to 0 or 1 values """
class BinaryMask:
    def __call__(self, tensor):
        tensor[tensor < MAX_VALUE / 2] = 0
        tensor[tensor >= MAX_VALUE / 2] = 1
        return tensor

""" Converts numpy mask to tensor """
class NumpyToTensor:
    def __call__(self, image):
        return torch.from_numpy(image).float().unsqueeze(0)

""" Data set of pacthed arial images for a given path, indices, patch size, overlap and image augmentation. """
class DatasetPatched(Dataset):
    def __init__(self, imgsPath, groundTruthPath, training_size, overlap_amount, rotation):
        imgs, groundtruths = read_rotate_patch(imgsPath, groundTruthPath, training_size, overlap_amount, rotation)
        self.imgsPatches = imgs
        self.groundTruth = groundtruths
        self.labels_transform = Compose([NumpyToTensor(), BinaryMask()])

    def __getitem__(self, index):
        label = img_as_ubyte(self.groundTruth[index])
        image = np.transpose(self.imgsPatches[index], (2, 0, 1))

        image = torch.from_numpy(image)
        if self.labels_transform != None:
            label = self.labels_transform(label)

        return image.float(), label

    def __len__(self):
        return len(self.imgsPatches)
