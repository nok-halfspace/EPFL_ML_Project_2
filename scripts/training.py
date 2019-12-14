import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import os
import psutil

from constants import *


# Chosen score : F1 metrics to be in accordance with AIcrowd

# def score(y_true,y_pred_onehot):

#     softMax = torch.nn.Softmax(1)
#     y_pred = torch.argmax(softMax(y_pred_onehot),1)

#     y_true_0 = y_true == False
#     y_pred_0 = y_pred == False

#     y_true_1 = y_true == True
#     y_pred_1 = y_pred == True

#     true_negative_nb = len(y_true[y_true_0 & y_pred_0])
#     false_negative_nb = len(y_true[y_true_1 & y_pred_0])
#     true_positive_nb = len(y_true[y_true_1 & y_pred_1])
#     false_positive_nb = len(y_true[y_true_0 & y_pred_1])

#     try:
#         precision = true_positive_nb / (true_positive_nb + false_positive_nb)
#         recall = true_positive_nb / (true_positive_nb + false_negative_nb)
#     except Exception as e:
#         precision = 0.5
#         recall = 0.5
#     f1 = 2 * (precision * recall) / (precision + recall + 0.01)
#     return f1

def score(y_true, y_pred_onehot):
    softMax = torch.nn.Softmax(1)
    y_pred_bin = torch.argmax(softMax(y_pred_onehot),1).view(-1)
    y_true = y_true.view(-1)
    f1 = f1_score(y_true.cpu(), y_pred_bin.cpu())
    return(f1)

def split_data(x,y,ratio, seed = 1):
    print(x.shape)
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


def training(model, loss_function, optimizer, x, y, epochs, ratio):
    val_loss_hist = []
    val_acc_hist = []
    train_acc_hist = []
    train_loss_hist = []

    x,val_x,y, val_y = split_data(x, y, ratio)
    process = psutil.Process(os.getpid())

    for epoch in range(epochs):
        model.train()

        print("Training, epoch=", epoch)
        print("Memory usage {0:.2f} GB".format(process.memory_info().rss/1024/1024/1024))
        loss_value = 0.0
        correct = 0
        for i in range(0,x.shape[0],BATCH_SIZE):
            data_inputs = x[i:BATCH_SIZE+i].to(DEVICE)
            data_targets = y[i:BATCH_SIZE+i].to(DEVICE)
            
            # HERE : Do data augmentation 

            print("Training image ", str(i))
            print("Memory usage {0:.2f} GB".format(process.memory_info().rss/1024/1024/1024))

            #Traning step
            optimizer.zero_grad()
            outputs = model(data_inputs)
            loss = loss_function(outputs, data_targets)
            loss.backward()
            optimizer.step()

            #Log
            loss_value += loss.item()
            correct += score(data_targets,outputs)

        loss_value /= x.shape[0]
        accuracy = correct/x.shape[0]

        #Validation prediction

        model.eval()
        loss_val_value = 0.0
        correct_val = 0
        for i in range(0,val_x.shape[0],BATCH_SIZE):
<<<<<<< HEAD
                
            data_val_inputs = val_x[i:BATCH_SIZE+i].to(DEVICE)
            data_val_targets = val_y[i:BATCH_SIZE+i].to(DEVICE)
            
            with torch.no_grad():
        
                outputs_val = model(data_val_inputs)
                val_loss = loss_function(outputs_val,data_val_targets)
            
             # log 
=======


            data_val_inputs = val_x[i:BATCH_SIZE+i].to(DEVICE)
            data_val_targets = val_y[i:BATCH_SIZE+i].to(DEVICE)

            outputs_val = model(data_val_inputs)
            val_loss = loss_function(outputs_val,data_val_targets)

             # log
>>>>>>> b52b01202a9f0398d4143ecc5e230498458a700e
            loss_val_value +=val_loss.item()
            correct_val += score(data_val_targets,outputs_val)


        loss_val_value /= val_x.shape[0]
        accuracy_val = correct_val/val_x.shape[0]

        #Log
        val_loss_hist.append(loss_val_value)
        val_acc_hist.append(accuracy_val)
        train_loss_hist.append(loss_value)
        train_acc_hist.append(accuracy)

        if DISPLAY:
            print(f'Epoch {epoch}, loss: {loss_value:.5f}, accuracy: {accuracy:.3f}, Val_loss: {loss_val_value:.5f}, Val_acc: {accuracy_val:.3f}')

        print(">> Saving Model for Epoch ", str(epoch))
        checkpoint = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
        torch.save(checkpoint, './model')

    return val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist


#Plot the logs of the loss and accuracy on the train/validation set
def plot_hist(val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))

    ax1.set_ylabel('Loss')
    ax2.set_ylabel('accuracy')

    ax1.plot(train_loss_hist,label='training')
    ax1.plot(val_loss_hist,label='validation')
    ax1.set_yscale('log')
    ax1.legend()

    ax2.plot(train_acc_hist,label='training')
    ax2.plot(val_acc_hist,label='validation')
    ax2.legend()

    plt.show()
