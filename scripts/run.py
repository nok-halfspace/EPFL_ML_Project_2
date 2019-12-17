import torch
from models import *
from testing import *
from preprocessing import *
from training import training
from constants import *
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import numpy as np

# TODO change these
from aerial_dataset import AerialDataset
from patched_aerial_dataset import PatchedAerialDataset
from preprocessing import *
from helpers import *
from mask_to_submission import *


def Usage():
    print("Usage\n\tpython3 run.py [train/predict]")
    exit(1)

def main():

    # Safety checkpoint
    if (len(sys.argv) < 2 or len(sys.argv) > 3):
        print('Number of arguments is incorrect')
        Usage()


    ''' Creating the Model '''
    network, criterion, optimizer = create_smaller_UNET() # 3 layers

    ''' Reloading an old model if user defines so ''' # <------- didn't test it recently

    if (sys.argv[1] == 'predict'):
        network.load_state_dict(torch.load(MODEL_PATH).state_dict())
    elif (sys.argv[1] == 'train'): # train
        # TODO: optional, add: ReduceLROnPlateau
        # TODO: optional, put early stopping
        ''' Read training images '''
        print(TRAINING_SIZE, VAL_SIZE)
        trainset = PatchedAerialDataset(TRAIN_IMAGE_PATH, TRAIN_GROUNDTRUTH_PATH, int(TRAINING_SIZE), PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT, ROTATION, ROTATION_ANGLES)  # TODO change this
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

        # to do : find a consistent number of images for validation
        valset = PatchedAerialDataset(TRAIN_IMAGE_PATH, TRAIN_GROUNDTRUTH_PATH, int(VAL_SIZE), PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT, ROTATION, ROTATION_ANGLES)  # TODO change this
        valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)


        ''' Training phase '''
        val_loss_hist, val_loss_hist_std, train_loss_hist, train_loss_hist_std, val_acc_hist, val_acc_hist_std, train_acc_hist, train_acc_hist_std = training(network, criterion, optimizer, score, trainloader, valloader, PATCH_SIZE, NUM_EPOCHS)
        torch.save(network, MODEL_PATH)
    else:
        print("Argument not recognized")
        Usage()

    # Load testing data
    testset = AerialDataset(TEST_IMAGE_PATH, NR_TEST_IMAGES)
    loader_test = DataLoader(testset, batch_size=1, shuffle=False) # To do : increase the batch_size
    # Predict labels
    roadsPredicted = predict_test_images(network, loader_test)

    # Transform pixel-wise prediction to patchwise
    patched_images = patch_prediction(roadsPredicted, TEST_IMG_SIZE, IMG_PATCH_SIZE)

    ''' Get patches for submission '''
    patches = []
    for im in patched_images:
        patches.extend(img_crop(im, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

    # Generate submission
    generate_predictions(NR_TEST_IMAGES, TEST_IMG_SIZE, IMG_PATCH_SIZE, patches, PREDICTED_PATH)
    submission_to_csv(SUBMISSION_PATH, PREDICTED_PATH)

    print('Done')
    print('Predictions generated in: {}'.format(PREDICTED_PATH))
    print('CSV submission generated in: {}'.format(SUBMISSION_PATH))

    # # TODO: We should use this part to create a submission
    # ''' Predicting on the test images '''
    # filenames_list = test_and_save_predictions(model, test_imgs)
    #
    # ''' create csv files '''
    # masks_to_submission(submissionFileName, filenames_list)

    # ''' create label images from csv '''
    # for i in range(1, NR_TEST_IMAGES+1):
    #     reconstruct_from_labels(i, submissionFileName)


main()
