import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing import *

def prepare_test_images(images_path, nr_test_images):
    # Load images
    print("nr_test_images=", nr_test_images)
    images = np.array(read_images(images_path, nr_test_images, "test_"))
    return images


class AerialDataset(Dataset):
    def __init__(self, images_path, nr_test_images):
        # Load images
        self.images = prepare_test_images(images_path, nr_test_images)

        # Transormation applied before getting an element
        self.images_transform = torch.from_numpy

    def __getitem__(self, index):
        image = np.transpose(self.images[index], (2, 0, 1))

        if self.images_transform != None:
            image = self.images_transform(image)

        return image.float()

    def __len__(self):
        return len(self.images)
