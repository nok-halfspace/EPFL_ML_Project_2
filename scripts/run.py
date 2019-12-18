import torch
from training import training
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import *
from constants import *
from models import *
from testing import *
from prepareData import *
from mask_to_submission import *
from DatasetPatch import *
from testDataset import *
from prepareData import *


def Usage():
    print("Usage\n\tpython3 run.py [--train/--predict]")
    exit(1)


def main():

    # Safety checkpoint
    if (len(sys.argv) < 2 or len(sys.argv) > 3):
        print('Number of arguments is incorrect')
        Usage()

    ''' Creating the Model '''
    network, criterion, optimizer = create_smaller_UNET() # 3 layers model

    if (sys.argv[1] == '--predict'):
        print("Loading the model... ")
        checkpoint = torch.load(MODEL_PATH)
        network.load_state_dict(checkpoint['state_dict'])

    elif (sys.argv[1] == '--train'):
        print("Reading", TRAINING_SIZE, "training image(s)")
        trainset = DatasetPatched(TRAIN_IMAGE_PATH, TRAIN_GROUNDTRUTH_PATH, TRAINING_SIZE, OVERLAY_SIZE, ROTATION)
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

        print("Reading", VAL_SIZE, "validation image(s)")
        valset = DatasetPatched(TRAIN_IMAGE_PATH, TRAIN_GROUNDTRUTH_PATH, VAL_SIZE, OVERLAY_SIZE, ROTATION)
        valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)

        ''' Training phase '''
        # TODO: optional, add: ReduceLROnPlateau
        # TODO: optional, put early stopping
        val_loss_hist, val_loss_hist_std, train_loss_hist, train_loss_hist_std, val_acc_hist, val_acc_hist_std, train_acc_hist, train_acc_hist_std = training(network, criterion, optimizer, score, trainloader, valloader, PATCH_SIZE, NUM_EPOCHS)
        print("Saving the model... ")
        torch.save({'state_dict': network.state_dict()}, MODEL_PATH)
    else:
        print("Argument not recognized")
        Usage()

    print("Load testing data... ")
    testset = TestDataset(TEST_IMAGE_PATH, NR_TEST_IMAGES)
    loader_test = DataLoader(testset, batch_size=1, shuffle=False) # To do : increase the batch_size

    print("Predict labels... ")
    roadsPredicted = predict_test_images(network, loader_test)

    # Transform pixel-wise prediction to patchwise
    patched_images = patch_prediction(roadsPredicted, TEST_IMG_SIZE, IMG_PATCH_SIZE)

    ''' Get patches for submission '''
    patches = getPatches(patched_images)

    ''' Generate submission '''
    reconstruct_img(NR_TEST_IMAGES, TEST_IMG_SIZE, IMG_PATCH_SIZE, patches, PREDICTED_PATH)
    submission_to_csv(SUBMISSION_PATH, PREDICTED_PATH)

    print('See latest submission file ', SUBMISSION_PATH)
    print('See predicted images at    ', PREDICTED_PATH)

main()
