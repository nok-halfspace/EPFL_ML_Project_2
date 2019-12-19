import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from sklearn.metrics import f1_score
import time
from utils import *
from constants import *


def score(labels, outputs):
    predictions = probability_to_prediction(outputs).flatten()
    labels = labels.squeeze().cpu().numpy().flatten()
    f1 = f1_score(labels, predictions)
    return f1


''' Training function '''
#def training(num_epochs, model, criterion, optimizer, trainset, trainloader, patch_size):
def training(model, criterion, optimizer, score, trainloader, valloader, patch_size, num_epochs):
    # Log of the losses and scores
    val_loss_hist, val_loss_hist_std, val_f1_hist, val_f1_hist_std = [], [], [], []
    train_f1_hist, train_f1_hist_std, train_loss_hist, train_loss_hist_std = [], [], [], []

    timestamp = time.strftime("%m:%d:%Y-%H:%M")
    f = open("logfile_" + timestamp + ".txt", 'w')

    print('Training the model...')
    for epoch in range(num_epochs):
        loss_value = []
        correct = []

        loss_value_val = []
        correct_val = []

        model.train()
        step = 1

        for data in trainloader:
            print("Epoch:", epoch+1, "/", num_epochs, " - Step", step, "/", len(trainloader))
            step += 1

            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # Training step
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value.append(loss.item())
            correct.append(score(labels, outputs))

        loss_value, loss_value_std = np.mean(loss_value), np.std(loss_value)
        f1,f1_std = np.mean(correct), np.std(correct)

        train_loss_hist.append(loss_value)
        train_loss_hist_std.append(loss_value_std)

        train_f1_hist.append(f1)
        train_f1_hist_std.append(f1_std)

        # Validation part
        print("Validation at Epoch:", epoch + 1, "/", num_epochs)
        model.eval()
        for data in valloader:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_value_val.append(loss.item())
                correct_val.append(score(labels, outputs))

        loss_validation, loss_validation_std = np.mean(loss_value_val), np.std(loss_value_val)
        f1_validation, f1_validation_std = np.mean(correct_val), np.std(correct_val)


        val_loss_hist.append(loss_validation)
        val_loss_hist_std.append(loss_validation_std)
        val_f1_hist.append(f1_validation)
        val_f1_hist_std.append(f1_validation_std)

        # Print at each epoch the evolution of quantities
        print('Epoch {} \n \
                Train loss: {} +/- {} \n \
                Train F1: {} +/- {} \n \
                Validation loss: {} +/- {} \n \
                Validation F1: {} +/- {} \
                '.format(epoch, loss_value, loss_value_std, f1, f1_std, loss_validation, loss_validation_std, f1_validation, f1_validation_std),
                    file=f)

    return val_loss_hist, val_loss_hist_std, train_loss_hist, train_loss_hist_std, val_f1_hist, val_f1_hist_std, train_f1_hist, train_f1_hist_std
