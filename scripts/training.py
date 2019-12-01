import torch
import matplotlib.pyplot as plt
import numpy as np 

from constants import *

# TO DO : define a score 

def split_data(x,y,ratio, seed = 1):
    
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(x.shape[0])
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_val = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_val = x[index_val]
    y_tr = y[index_tr]
    y_val = y[index_val]
    return x_tr, x_val, y_tr, y_val
            


def training(network, loss_function, optimizer, score, x, y, epochs, ratio=0.2):
    val_loss_hist = []
    val_acc_hist = []
    train_acc_hist = []
    train_loss_hist = []
    
    x,y,val_x,val_y = split_data(x, y, ratio)
    
    for epoch in range(epochs):
        loss_value = 0.0
        correct = 0
        for i in range(0,x.shape[0],BATCH_SIZE):
            data_inputs = x[i:BATCH_SIZE+i]
            data_targets = y[i:BATCH_SIZE+i]

            #Traning step
            optimizer.zero_grad()
            outputs = network(data_inputs)
            loss = loss_function(outputs, data_targets)
            loss.backward()
            optimizer.step()

            #Log
            loss_value += loss.item()
            correct += score(data_targets,outputs) 

        loss_value /= x.shape[0]
        accuracy = correct/x.shape[0]
        
        #Validation prediction
        outputs = network(val_x)
        val_loss = loss_function(outputs,val_y) / val_x.shape[0]
        val_acc = score(val_y,outputs)/val_x.shape[0]
        
        #Log
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(loss_value)
        train_acc_hist.append(accuracy)
        
        if DISPLAY:
            print(f'Epoch {epoch}, loss: {loss_value:.5f}, accuracy: {accuracy:.3f}, Val_loss: {val_loss:.5f}, Val_acc: {val_acc:.3f}')
    
    return val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist


#Plot the logs of the loss and accuracy on the train/validation set
def plot_hist(val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('accuracy')
    
    ax1.plot(train_loss_hist,label='trainining')
    ax1.plot(val_loss_hist,label='validation')
    ax1.set_yscale('log')
    ax1.legend()
    
    ax2.plot(train_acc_hist,label='training')
    ax2.plot(val_acc_hist,label='validation')
    ax2.legend()
    
    plt.show()