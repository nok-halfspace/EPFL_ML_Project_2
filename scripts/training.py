import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import os
import psutil
from scipy import ndimage
from PIL import Image
import scipy.misc
from constants import *
import numpy as np


# Chosen score : F1 metrics to be in accordance with AIcrowd

def score(y_true, y_pred_onehot):
    softMax = torch.nn.Softmax(1)
    y_pred_bin = torch.argmax(softMax(y_pred_onehot),1).view(-1)
    y_true = y_true.view(-1)
    f1 = f1_score(y_true.cpu(), y_pred_bin.cpu())
    return(f1)

def split_data(x,y,ratio, seed = 1):

    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_img = x.shape[0]
    indices = np.random.permutation(num_img)
    index_split = int(np.floor(ratio * num_img))
    index_tr = indices[: index_split]
    index_val = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_val = x[index_val]
    y_tr = y[index_tr]
    y_val = y[index_val]
    return x_tr, x_val, y_tr, y_val


''' Training function '''
def training(num_epochs, model, criterion, optimizer, trainset, trainloader, patch_size):
    best_model_wts = model.state_dict()

    print('Training the model...')
    for epoch in range(num_epochs):
        epoch_loss_train = []
        model.train()

        step = 1
        for data in trainloader:
            print("Epoch:", epoch+1, "/", num_epochs, " - Step", step, "/", len(trainloader))
            step += 1

            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss_train.append(loss.data.item())


    return best_model_wts
