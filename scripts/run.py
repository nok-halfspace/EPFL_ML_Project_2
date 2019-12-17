import torch
import torch.nn as nn
from models2 import *
from testing import *
from preprocessing import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from training import training
from constants import *
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import numpy as np

# TODO change these
from aerial_dataset import AerialDataset
from patched_aerial_dataset import PatchedAerialDataset
from preprocessing import *
from visualization.helpers import *
from mask_to_submission import *


def main():

    ''' Read training images '''
    trainset = PatchedAerialDataset(TRAIN_IMAGE_PATH, TRAIN_GROUNDTRUTH_PATH, TRAINING_SIZE, PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT, ROTATION, ROTATION_ANGLES)  # TODO change this
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    ''' Creating the Model '''
    network, criterion, optimizer = create_UNET() # 3 layers

    ''' Reloading an old model if user defines so '''
    if (RELOAD_MODEL == True):
        print("Reloading the model from the disk...")
        checkpoint = torch.load(MODEL_PATH)
        network.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # TODO: optional, add: ReduceLROnPlateau
    # TODO: optional, put early stopping

    # summary(model, input_size=(3,400,400)) # prints memory resources

    ''' Training phase '''
    best_model_wts = training(NUM_EPOCHS, network, criterion, optimizer, trainset, trainloader, PATCH_SIZE)

    # Load testing data
    testset = AerialDataset(TEST_IMAGE_PATH, NR_TEST_IMAGES)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Predict labels
    roadsPredicted = predict(network, testloader)

    # Transform pixel-wise prediction to patchwise
    patched_images = [labels_to_patches(labels, TEST_IMG_SIZE, IMG_PATCH_SIZE, 0.25) for labels in roadsPredicted]  # TODO: Change this

    ''' Get patches for submission '''
    patches = []
    for im in patched_images:
        patches.extend(img_crop(im, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

    # Generate submission
    generate_predictions(NR_TEST_IMAGES, TEST_IMG_SIZE, IMG_PATCH_SIZE, patches, PREDICTED_PATH)
    generate_submission_csv(SUBMISSION_PATH, PREDICTED_PATH)

    print('Done')
    print('Predictions generated in: {}'.format(PREDICTED_PATH))
    print('CSV submission generated in: {}'.format(SUBMISSION_PATH))

    # # TODO: We should use this part to create a submission
    # ''' Predicting on the test images '''
    # filenames_list = test_and_save_predictions(model, test_imgs)
    #
    # ''' create csv files '''
    # masks_to_submission(submissionFileName, filenames_list)
    #
    # ''' create label images from csv '''
    # for i in range(1, NR_TEST_IMAGES+1):
    #     reconstruct_from_labels(i, submissionFileName)


main()
