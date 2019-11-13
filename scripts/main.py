import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import os

TRAINING_SIZE = 20
NUM_EPOCHS = 100

def main():

    to_tensor = transforms.ToTensor() #ToTensor transforms the image to a tensor with range [0,1]
    num_images = TRAINING_SIZE

    data_dir = '../Datasets/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'    

    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            t_img = to_tensor(img).unsqueeze(0) # 1 x 3 x 400 x 400
            imgs.append(t_img)
    print(imgs)

from tf_aerial_images import extract_data, extract_labels

def main2():


    data_dir = '/Users/natashaklingenbrunn/Desktop/ML_course/CS-433-Project-2/Datasets/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE) # [image, y, x, [r, g, b]]
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE) # [image index, label index]

    num_epochs = NUM_EPOCHS

    c0 = 0  # background
    c1 = 0  # road

    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 += 1
        else:
            c1 += 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')

    min_c = min(c0, c1) # find the smaller class

    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1] #[1, 0]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1] #[0, 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]

    print(train_data.shape)
    
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0

    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

main()