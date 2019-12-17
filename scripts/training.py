import torch
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import numpy as np


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
