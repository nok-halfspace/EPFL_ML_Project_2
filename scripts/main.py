import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import os

TRAINING_SIZE = 20
NUM_EPOCHS = 100

def extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename):

    to_tensor = transforms.ToTensor() #ToTensor transforms the image to a tensor with range [0,1]
    num_images = TRAINING_SIZE

    imgs = []

    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            t_img = to_tensor(img).unsqueeze(0) # 1 x 3 [rgb] x 400 x 400
            imgs.append(t_img)
    return imgs # length TRAINING_SIZE

def main():

    data_dir = '../Datasets/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'    

    imgs = extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename)
    
