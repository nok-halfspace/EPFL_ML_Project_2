import torch
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import numpy as np
from torch.utils.data import random_split


''' Training function '''
#def training(num_epochs, model, criterion, optimizer, trainset, trainloader, patch_size):
def training(model, criterion, optimizer, score, trainloader, valloader, patch_size, num_epochs):
    # Log of the losses and scores 
    val_loss_hist = []
    val_loss_hist_std = []
    val_acc_hist = []
    val_acc_hist_std = []
    
    train_acc_hist = []
    train_acc_hist_std = []
    train_loss_hist = []
    train_loss_hist_std = []
    
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
            correct.append(score(labels, outputs)) # Get a normalized score over the batch size  # In score => put the prediction 0 or 1
        
        loss_value_epoch, loss_value_epoch_std = np.mean(loss_value), np.std(loss_value)      
        accuracy_epoch, accuracy_epoch_std = np.mean(correct), np.std(correct)
        
        train_loss_hist.append(loss_value_epoch)
        train_loss_hist_std.append(loss_value_epoch_std)
        
        train_acc_hist.append(accuracy_epoch)
        train_acc_hist_std.append(accuracy_epoch_std)   
        
        # Validation part 
        
        print("Validation at Epoch:", epoch+1, "/", num_epochs)  
        model.eval()
        for data in valloader:           
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
          
                with torch.no_grad() :
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            
                    loss_value_val.append(loss.item())
                    correct_val.append(score(labels, outputs))
       
        loss_value_val_epoch, loss_value_val_epoch_std = np.mean(loss_value_val), np.std(loss_value_val)      
        accuracy_val_epoch, accuracy_val_epoch_std = np.mean(correct_val), np.std(correct_val)
        
        
        val_loss_hist.append(loss_value_val_epoch)
        val_loss_hist_std.append(loss_value_val_epoch_std)
        val_acc_hist.append(accuracy_val_epoch)
        val_acc_hist_std.append(accuracy_val_epoch_std)
        
        # Print at each epoch the evolution of quantities
        
        print(f'Epoch {epoch}, loss: {loss_value_epoch:.5f} +/- {test_std:.5f}, accuracy: {accuracy_epoch:.5f} +/- {accuracy_epoch_std:.5f}, \n Val_loss: {loss_value_val_epoch:.5f} +/- {loss_value_val_epoch_std:.5f} , Val_acc: {accuracy_val_epoch:.5f} +/- {accuracy_val_epoch_std:.5f} ')
        
        

    return val_loss_hist, val_loss_hist_std, train_loss_hist, train_loss_hist_std, val_acc_hist, val_acc_hist_std, train_acc_hist, train_acc_hist_std
