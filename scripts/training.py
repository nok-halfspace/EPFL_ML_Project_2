import torch
import matplotlib.pyplot as plt
import numpy as np
from constants import *
import numpy as np
from torch.utils.data import random_split


''' Training function '''
#def training(num_epochs, model, criterion, optimizer, trainset, trainloader, patch_size):
def training(model, criterion, optimizer, score, trainloader, valloader, patch_size, num_epochs):
    best_model_wts = model.state_dict() # Not sure if really needed ? 
    
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
                
        train_loss_hist.append(np.mean(loss_value))
        train_loss_hist_std.append(np.std(loss_value))
        train_acc_hist.append(np.mean(correct))
        train_acc_hist_std.append(np.std(correct))
        
        
        # Validation step every ten epochs 
        if (epoch % 10 == 0) :
            print("Validation at Epoch:", epoch+1, "/", num_epochs)  
            model.eval()
            for data in valloader:           
            
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            
            # Training step 
                with torch.no_grad() :
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            
                    loss_value_val.append(loss.item())
                    correct_val.append(score(labels, outputs))
       
        
        val_loss_hist.append(np.mean(loss_value_val))
        val_loss_hist_std.append(np.std(loss_value_val))
        val_acc_hist.append(np.mean(correct_val))
        val_acc_hist_std.append(np.std(correct_val))
        
        

    return best_model_wts, val_loss_hist, val_loss_hist_std, train_loss_hist, train_loss_hist_std, val_acc_hist, val_acc_hist_std, train_acc_hist, train_acc_hist_std
