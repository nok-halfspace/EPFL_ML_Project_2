import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import sys
from models import *
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from training import training
import sklearn.metrics as metrics
import constants

'''
Everyone: Refresh entire code, add comments !
          Data Augmentation Procedures

Clara: f1-score
Natasha: converting to submission file
Daniel: uploading to cloud to compute

'''

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

def extract_feature_vectors(TRAINING_SIZE, data_dir, path, rotate = True, save = False):
    train_data_filename = data_dir + path
    to_tensor = transforms.ToTensor() #ToTensor transforms the image to a tensor with range [0,1]
    num_images = TRAINING_SIZE
    imgs = []
    r_imgs = []

    for i in range(1, num_images+1):

        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        print(image_filename)
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            t_img = to_tensor(img) #3 [rgb] x 400 x 400
            imgs.append(t_img)

            if (rotate == True):
                width, height = img.size
                # Rotate images

                for degree in range(360):
                    rotated_img = img.rotate(degree)
                    # Crop to remove black parts
                    left     = width / 4 * 0.58
                    top      = height / 4 * 0.58
                    right    = width - left
                    bottom   = height - top
                    rotated_img = rotated_img.crop((left, top, right, bottom))
                    if (save == True):
                        # optional, save new image on disk, to see the effect
                        rotated_img.save("../Rotations/"+path + imageid + "_"+str(degree)+".png")
                    rt_img = to_tensor(rotated_img) # 3 [rgb] x 284 x 284
                    r_imgs.append(rt_img)
        else:
            print(image_filename, "is not a file, follow the README instruction to run the project. (check path)", file=sys.stderr)
            sys.exit()
    imgs, r_imgs = torch.stack(imgs), torch.stack(r_imgs)
    return imgs, r_imgs

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
    train_data_filename = 'images/'
    train_labels_filename = 'groundtruth/'

    imgs, r_imgs =  extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename, True)
    labels, r_labels = extract_feature_vectors(TRAINING_SIZE, data_dir, train_labels_filename, True)

    labels = F.pad(labels, (2, 2, 2, 2), mode = 'reflect') # to get a label vector of the same size as our network's ouput
    labels_onehot = [value_to_class(labels[i]) for i in range(TRAINING_SIZE)] # one hot output
    labels_onehot = torch.stack(labels_onehot) # torch object of list

    input = torch.Tensor(imgs[:][0])

    model, loss, optimizer = create_UNET()
    outputs = model(imgs)

    outputs = outputs.view(TRAINING_SIZE, N_CLASSES, -1)
    val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = training(model, loss, optimizer, imgs, labels_onehot, epochs, ratio=0.5)

    return val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist


if __name__== "__main__":
       val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = main()
       print(val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist)
