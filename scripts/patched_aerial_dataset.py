import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from skimage import img_as_ubyte
from preprocessing import *

MAX_VALUE = 256

class Relabel:
    """
    Transforms grayscale image label to binary.
    """
    def __call__(self, tensor):
        assert isinstance(tensor, torch.FloatTensor), 'tensor needs to be FloatTensor'
        tensor[tensor < MAX_VALUE / 2] = 0
        tensor[tensor >= MAX_VALUE / 2] = 1
        return tensor


class ToLabel:
    """
    Transforms numpy labels to pytorch tensors.
    """
    def __call__(self, image):
        return torch.from_numpy(image).float().unsqueeze(0)


class PatchedAerialDataset(Dataset):
    """
    Data set of pacthed arial images for a given path, indices, patch size, overlap and image augmentation.
    """
    def __init__(self, images_path, labels_path, indices, patch_size, overlap, overlap_amount, rotation, rotation_angles):
        # Load images and labels
        images, labels = prepare_train_patches(images_path, labels_path, indices, patch_size, overlap, overlap_amount, rotation, rotation_angles)
        self.patched_images = images
        self.patched_labels = labels

        # Define out transformations
        self.images_transform = torch.from_numpy
        self.labels_transform = Compose([ToLabel(), Relabel()])

    def __getitem__(self, index):
        patched_image = np.transpose(self.patched_images[index], (2, 0, 1))
        patched_label = img_as_ubyte(self.patched_labels[index])

        if self.images_transform != None:
            patched_image = self.images_transform(patched_image)
        if self.labels_transform != None:
            patched_label = self.labels_transform(patched_label)

        return patched_image.float(), patched_label

    def __len__(self):
        return len(self.patched_images)
