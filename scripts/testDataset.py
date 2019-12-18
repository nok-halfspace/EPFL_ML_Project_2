import torch
from torch.utils.data import Dataset
import numpy as np
from prepareData import read_images


def read_test_images(images_path, nr_test_images):
    images = np.array(read_images(images_path, nr_test_images, "test_"))
    return images


class TestDataset(Dataset):
    def __init__(self, images_path, nr_test_images):
        self.testImages = read_test_images(images_path, nr_test_images)

    def __getitem__(self, index):
        image = np.transpose(self.testImages[index], (2, 0, 1))
        image = torch.from_numpy(image)
        return image.float()

    def __len__(self):
        return len(self.testImages)
