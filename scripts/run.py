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
from torchsummary import summary

def readTrainingImages(TRAINING_SIZE, data_dir, path, rotate = False, save = False):
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

""" Reading test images """

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
    # can be improved using scatter probably
    # or see how it is done in the tf_aerial.py
def value_to_class(img):
    img = img.squeeze()
    H = img.shape[0]
    W = img.shape[1]
    labels = torch.randn((H,W)).to(DEVICE)
    foreground_threshold = 0.5
    for h in range(H) :
        for w in range(W) :
            if img[h,w] > foreground_threshold:  # road
                labels[h,w] = torch.tensor(1.0)
            else:  # bgrd
                labels[h,w] = torch.tensor(0.0)
    return labels.long()


def main():

    # process = psutil.Process(os.getpid()) ## in case we need to verify memory usage
    # print(process.memory_info().rss/1024/1024)  # in Mbytes

    # Reading test images
    test_imgs = readTestImages(test_dir, NR_TEST_IMAGES)

    # Reading training images
    train_imgs, r_imgs = readTrainingImages(TRAINING_SIZE, data_dir, train_data_filename, rotateFlag) # satellite
    labels, r_labels = readTrainingImages(TRAINING_SIZE, data_dir, train_labels_filename, rotateFlag) # labels

#     # Preprocessing
    labels = F.pad(labels, (2, 2, 2, 2), mode = 'reflect') # to get a label vector of the same size as our network's output
    labels_bin =  torch.stack([value_to_class(labels[i]) for i in range(TRAINING_SIZE)]) # decimal to binary

    # Creating the outline of the model we want
    # model, loss, optimizer = create_UNET() # 5 layers

    model = create_UNET() # 5 layer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    if (RELOAD_MODEL == True):
        print("Reloading the model from the disk...")
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    model.eval()

    summary(model, input_size=(3,400,400)) # prints memory resources

    # Training all (TRAINING_SIZE) images
    val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = training(model, loss, optimizer, train_imgs, labels_bin, NUM_EPOCHS, RATIO)

    # Predicting on thw the test images
    filenames_list = test_and_save_predictions(model, test_imgs)

    # Create csv files
    masks_to_submission(submissionFileName, filenames_list)

    # create label images from csv
    for i in range(1, NR_TEST_IMAGES+1):
        reconstruct_from_labels(i, submissionFileName)

    return val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist



if __name__== "__main__":
        val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = main()
        print("Training loss = ", train_loss_hist)
        print("Testing loss = ", val_loss_hist)
        print("Training accuracy = ", train_acc_hist)
        print("Testing accuracy = ", val_acc_hist)
