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
from visualization.helpers import labels_to_patches, extract_patches, generate_predictions
from mask_to_submission import *


def main():

    # process = psutil.Process(os.getpid()) ## in case we need to verify memory usage
    # print(process.memory_info().rss/1024/1024)  # in Mbytes

    # Get augmentation configuration
    aug_config = ImageAugmentationConfig()
    aug_config.rotation([45, 90, 135, 180, 225, 270, 315])

    # Create dataset and dataloader
    indices = np.arange(1, TRAINING_SIZE + 1)  # TODO change this
    trainset = PatchedAerialDataset(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, indices, PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT, aug_config)  # TODO change this
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    ''' Creating the Model '''
    network, criterion, optimizer = create_UNET() # 3 layer

    ''' Reloading an old model if user defines so '''
    if (RELOAD_MODEL == True):
        print("Reloading the model from the disk...")
        checkpoint = torch.load(MODEL_PATH)
        network.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])


    slowDown = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)  # TODO: Check without it

    # TODO: put early stopping


    # summary(model, input_size=(3,400,400)) # prints memory resources
    # Train model
    datasets, dataloaders = {'train': trainset}, {'train': trainloader}
    scores, train_loss, val_loss, best_model_wts = training(NUM_EPOCHS, network, criterion, optimizer, slowDown, datasets, dataloaders, PATCH_SIZE, validate=False)
    print(scores, train_loss, val_loss)


    # val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = training(network, criterion, optimizer, train_imgs, labels_bin, NUM_EPOCHS, RATIO, slowDown)



    # Load testing data
    test_indices = np.arange(1, NR_TEST_IMAGES + 1)
    testset = AerialDataset(TEST_IMAGE_PATH, test_indices, aug_config, majority_voting=False) # TODO: Change this
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Predict labels
    predicted_labels = predict(network, testloader) # TODO: Change this

    # Transform pixel-wise prediction to patchwise
    patched_images = [labels_to_patches(labels, TEST_IMG_SIZE, TEST_PATCH_SIZE, 0.25) for labels in predicted_labels]  # TODO: Change this

    # Extract each patch
    img_patches_submit = extract_patches(patched_images, TEST_PATCH_SIZE)

    # Generate submission
    generate_predictions(NR_TEST_IMAGES, TEST_IMG_SIZE, TEST_PATCH_SIZE, img_patches_submit, TEST_LABEL_PATH)
    generate_submission_csv(SUBMISSION_PATH, TEST_LABEL_PATH)

    print('Done')
    print('Predictions generated in: {}'.format(TEST_LABEL_PATH))
    print('CSV submission generated in: {}'.format(SUBMISSION_PATH))

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
