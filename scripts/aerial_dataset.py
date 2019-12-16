"""
Full aerial images dataset.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing import *

def prepare_test_images(images_path, indices, aug_config, majority_voting=False):
    """
    Load and augment images for testing.
    Note that only channels augmentations are performed.
    """
    # Load images
    #NOTE: needed np.arrays to append that stuff easily
    images = np.array(extract_images(images_path, indices, "test_"))

    if majority_voting:
        #This has been chosen so that the structure of images is
        #50 images no rotate | 50 images rotate 90 | 50 images rotate 180 | 50 images rotate 270
        #instead of
        #img0 | img0 rotate 90 | img0 rotate 180 | img0 rotate 270 | img1 | img1 rotate 90 | ...
        #This is easier then to rotate back the images afters model eval
        r1 = np.array(rotate_images(images, [90]))
        r2 = np.array(rotate_images(images, [180]))
        r3 = np.array(rotate_images(images, [270]))
        images = np.concatenate((images, r1, r2, r3), axis = 0)

    # Augment channels if necessary
    if aug_config.augment_channels:
        images = augment_channels(images, aug_config)
    return images


class AerialDataset(Dataset):
    """
    Data set of arial images for a given path, indices and image augmentation.
    It is possible to use majority voting (allows to predict a pixel label by voting of the 4 main rotation predictions).
    """
    def __init__(self, images_path, indices, augmentation_config, majority_voting=False):
        # Load images
        self.images = prepare_test_images(images_path, indices, augmentation_config, majority_voting)

        # Transormation applied before getting an element
        self.images_transform = torch.from_numpy

    def __getitem__(self, index):
        image = np.transpose(self.images[index], (2, 0, 1))

        if self.images_transform != None:
            image = self.images_transform(image)

        return image.float()

    def __len__(self):
        return len(self.images)
