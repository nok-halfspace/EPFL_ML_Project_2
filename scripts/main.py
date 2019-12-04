import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import os
from models import * 
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as metrics

#TRAINING_SIZE = 20
TRAINING_SIZE = 1 # Debug purposes 
NUM_EPOCHS = 100
N_CLASSES = 2

# This function returns a list of patches from image (3D),
# each patch has a size of patch_h * patch_w
def getPatches(image, patch_h, patch_w):
    patches = []
    width = image.shape[1]
    height = image.shape[2]
    # print(width, height)
    for i in range(0, height, patch_h):
        for j in range(0, width, patch_w):
            patch = image[:, j:j+patch_w, i:i+patch_h]
            patches.append(patch)
    return patches

def extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename):

    to_tensor = transforms.ToTensor() #ToTensor transforms the image to a tensor with range [0,1]
    num_images = TRAINING_SIZE
    imgs = []

    for i in range(1, num_images+1):

        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        print(image_filename)
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)

            t_img = to_tensor(img) #  3 [rgb] x 400 x 400
            imgs.append(t_img)
    imgs = torch.stack(imgs)
    return imgs # length TRAINING_SIZE

    # Assign a one-hot label to each pixel of a ground_truth image
    # can be improved usign scatter probably
def value_to_class(img):
    img_labels = img.view(-1) # image to vector 
    n_pix = img_labels.shape[0]
    labels_onehot = torch.randn((N_CLASSES,n_pix))
    foreground_threshold = 0.5  
    for pix in range(n_pix) : 
        if img_labels[pix] > foreground_threshold:  # road
            labels_onehot[:,pix] = torch.tensor([0, 1])
        else:  # bgrd
            labels_onehot[:,pix] = torch.tensor([1, 0])
    return labels_onehot


def main():

    data_dir = '../Datasets/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'
    
    #Patches not needed anymore 
    #patch_h = 32
    #patch_w = 32


    imgs =  extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename)
    labels = extract_feature_vectors(TRAINING_SIZE, data_dir, train_labels_filename) 
    labels = F.pad(labels, (2, 2, 2, 2), mode = 'reflect') # to get a label vector of the same size as our network's ouput
    labels_onehot = [value_to_class(labels[i]) for i in range(TRAINING_SIZE)]
    labels_onehot = torch.stack(labels_onehot)


    model, loss, optimizer = create_UNET()
    outputs = model(imgs)
    
    return(outputs)
    
    val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = training(model, loss, optimizer, score, x, y, epochs, ratio=0.2)


if __name__== "__main__":
       main()
