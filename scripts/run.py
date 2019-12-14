

import torch
import torch.nn as nn
from models import *
from testing import *
from preprocessing import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from training import training
from constants import *
from torchsummary import summary




def main():

    # process = psutil.Process(os.getpid()) ## in case we need to verify memory usage
    # print(process.memory_info().rss/1024/1024)  # in Mbytes

    # Reading test images
    test_imgs = readTestImages(test_dir, NR_TEST_IMAGES)

    ''' Reading training images '''
    train_imgs, r_imgs = readTrainingImages(TRAINING_SIZE, data_dir, train_data_filename, rotateFlag) # groundtruth
    labels, r_labels = readTrainingImages(TRAINING_SIZE, data_dir, train_labels_filename, rotateFlag) # labels

    ''' Preprocessing, getting the labels via 16 x 16 patches of the raw input '''
    labels = F.pad(labels, (2, 2, 2, 2), mode = 'reflect') # to get a label vector of the same size as our network's output
    labels_bin =  torch.stack([value_to_label_by_patches(labels[i]) for i in range(TRAINING_SIZE)]) # decimal to binary

    # for image in labels_bin:
    #     plt.figure()
    #     plt.imshow(image.numpy(), cmap="gray")
    #     plt.show()
    # quit()

    ''' Creating the Model '''
    model = create_UNET() # 5 layer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    ''' Reloading an old model if user defines so '''
    if (RELOAD_MODEL == True):
        print("Reloading the model from the disk...")
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    ''' Training all (TRAINING_SIZE) images '''
    model.eval()
    # summary(model, input_size=(3,400,400)) # prints memory resources
    val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = training(model, loss, optimizer, train_imgs, labels_bin, NUM_EPOCHS, RATIO)

    ''' Predicting on the test images '''
    filenames_list = test_and_save_predictions(model, test_imgs)

    ''' create csv files '''
    masks_to_submission(submissionFileName, filenames_list)

    ''' create label images from csv '''
    for i in range(1, NR_TEST_IMAGES+1):
        reconstruct_from_labels(i, submissionFileName)

    print("Training loss = ", train_loss_hist)
    print("Testing loss = ", val_loss_hist)
    print("Training accuracy = ", train_acc_hist)
    print("Testing accuracy = ", val_acc_hist)


main()
