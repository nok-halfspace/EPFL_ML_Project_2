import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import psutil
import sys
from models import *
from testing import *
import torch.optim as optim
import torch.nn.functional as F
from training import training
from constants import *


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

    if rotate:
        r_imgs = torch.stack(r_imgs)
    imgs = torch.stack(imgs)

    return imgs, r_imgs


def readTestImages(test_directory, num_images):
    imgs = []
    to_tensor = transforms.ToTensor()
    current_image_path = ""
    for i in range(1, num_images+1):
        current_image_path = test_directory + str(i) + "/test_" + str(i) + ".png"
        if os.path.isfile(current_image_path):
            img = Image.open(current_image_path)
            t_img = to_tensor(img) # 3 [rgb] x 600 x 600
            imgs.append(t_img)

        else:
            print(current_image_path, "is not a file, follow the README instruction to run the project. (check path)", file=sys.stderr)
            sys.exit()

    imgs = torch.stack(imgs)
    return imgs



    # Assign a one-hot label to each pixel of a ground_truth image
    # can be improved usign scatter probably
    # or see how it is done in the tf_aerial.py
def value_to_class(img):
    img = img.squeeze()
    H = img.shape[0]
    W = img.shape[1]
    labels = torch.randn((H,W))
    foreground_threshold = 0.5
    for h in range(H) :
        for w in range(W) :
            if img[h,w] > foreground_threshold:  # road
                labels[h,w] = torch.tensor(1.0)
            else:  # bgrd
                labels[h,w] = torch.tensor(0.0)
    return labels.long()


def main():
    data_dir = '../Datasets/training/'
    test_dir = '../Datasets/test_set_images/test_'

    train_data_filename = 'images/'
    train_labels_filename = 'groundtruth/'
    rotateFlag = False
    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss/1024/1024)  # in Mbytes
    print("Reading test images...")
    test_imgs = readTestImages(test_dir, NR_TEST_IMAGES)

    imgs, r_imgs = extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename, rotateFlag)
    labels, r_labels = extract_feature_vectors(TRAINING_SIZE, data_dir, train_labels_filename, rotateFlag)

    labels = F.pad(labels, (2, 2, 2, 2), mode = 'reflect') # to get a label vector of the same size as our network's ouput

    labels_bin =  torch.stack([value_to_class(labels[i]) for i in range(TRAINING_SIZE)])
    print(labels_bin.type())

    epochs = NUM_EPOCHS
    model, loss, optimizer = create_UNET()
    val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = training(model, loss, optimizer, imgs, labels_bin, epochs, ratio=0.5)
    filenames_list = test_and_save_predictions(model, test_imgs)

    submissionFileName = "latestSubmission.csv"
    # Create submission file
    masks_to_submission(submissionFileName, filenames_list)

    # create label images from submissionFile
    for i in range(1, NR_TEST_IMAGES+1):
        reconstruct_from_labels(i, submissionFileName)

    return val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist


if __name__== "__main__":
        val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = main()
        print("Validation loss = ", val_loss_hist)
        print("Training loss = ", train_loss_hist)
        print("Validation accuracy = ", val_acc_hist)
        print("Training accuracy = ", train_acc_hist)
