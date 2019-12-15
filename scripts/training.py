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



def training(num_epochs, model, criterion, optimizer, lr_scheduler, datasets, dataloaders, patch_size, validate=True):
    """
    TODO
    """
    best_score = 0.0
    best_model_wts = model.state_dict()

    scores, train_loss, val_loss = [], [], []
    phases = ['train', 'val'] if validate else ['train']

    print('Starting training and validation of the model...')
    for epoch in range(num_epochs):
        epoch_loss_train, epoch_loss_val, f1score_sum = [], [], 0.0
        print("Epoch", epoch)
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            order = 0
            for data in dataloaders[phase]:
                print("Step", order, "/", len(dataloaders[phase]))
                order += 1
                # Load batch
                inputs, labels = data

                # # TODO Check with these lays uncommented
                # if cuda:
                #     inputs, labels = Variable(inputs.cuda(gpu_idx)), Variable(labels.cuda(gpu_idx))
                # else:
                #     inputs, labels = Variable(inputs), Variable(labels)

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Forward and compute loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss_train.append(loss.data.item())
                else:
                    # Convert float tensor to label prediction
                    preds_np = np.rint(outputs.squeeze().data.cpu().numpy())
                    # Convert float tensor labels to numpy labels
                    labels_np = labels.squeeze().data.cpu().numpy()

                    # Compare and compute f1-score
                    f1score_sum += f1_score(labels_np.ravel(), preds_np.ravel(), average='micro')
                    epoch_loss_val.append(loss.data.item())

        if validate:
            # Validate epoch results
            epoch_score = f1score_sum / len(datasets['val'])
            scores.append(epoch_score)
            train_loss.append(np.mean(epoch_loss_train))
            val_loss.append(np.mean(epoch_loss_val))
            progress_str = '[epoch {}/{}] - Valid Loss: {:.4f} Valid score: {:f}%'.format(epoch + 1, num_epochs, np.mean(epoch_loss_val), epoch_score*100)

            if epoch_score > best_score:
                best_score = epoch_score
                best_model_wts = model.state_dict()

        else:
            progress_str = '[epoch {}/{}]'.format(epoch + 1, num_epochs)

        print(progress_str)

        # Adjust learning rate
        lr_scheduler.step(int(np.mean(epoch_loss_train) * 1000))

    # Load best model if validate mode enabled
    if validate:
        model.load_state_dict(best_model_wts)

    return scores, train_loss, val_loss, best_model_wts

#
# def training(model, loss_function, optimizer, x, y, epochs, ratio):
#     val_loss_hist = []
#     val_acc_hist = []
#     train_acc_hist = []
#     train_loss_hist = []
#
#     x, val_x, y, val_y = split_data(x, y, ratio)
#     process = psutil.Process(os.getpid())
#
#     for epoch in range(epochs):
#
#         ''' Training '''
#
#         model.train()
#         loss_value = 0.0
#         correct = 0.0
#         for i in range(0, x.shape[0], BATCH_SIZE):
#
#             data_inputs = x[i:BATCH_SIZE+i].to(DEVICE)
#             data_targets = y[i:BATCH_SIZE+i].to(DEVICE)
#
#             #Traning step
#             optimizer.zero_grad()
#             outputs = model(data_inputs)
#             outputs = outputs[:,:,2:-2,2:-2]
#             loss = loss_function(outputs, data_targets)
#
#             correct += score(data_targets, outputs)
#
#             # HERE : Do data augmentation
#             data_input_numpy = data_inputs[0].cpu().numpy()
#             data_targets_numpy = data_targets.cpu().numpy()
#
#             # img = Image.fromarray(data_input_numpy.T, 'RGB') # if you want to check the initial image
#             # img.save('out.png')
#
#             print("Memory usage {0:.2f} GB".format(process.memory_info().rss/1024/1024/1024))
#
#             loss_rotated = 0
#             n_rotions = 17
#             for tetha in range(1, n_rotions+1):
#                 angle = 10 * tetha
#                 print("Rotating image", i," with ", angle, "degrees.")
#                 print("Memory usage {0:.2f} GB".format(process.memory_info().rss/1024/1024/1024))
#                 data_input_numpy_rotated = ndimage.rotate(data_input_numpy, angle, reshape=False, axes=(1,2), mode='reflect')
#                 data_input_rotated = torch.tensor(data_input_numpy_rotated)
#                 data_input_rotated = torch.unsqueeze(data_input_rotated, 0).to(DEVICE)
#                 data_target_numpy_rotated = ndimage.rotate(data_targets_numpy, angle, reshape=False, axes=(1,2), mode='reflect')
#                 data_target_rotated = torch.tensor(data_target_numpy_rotated).to(DEVICE)
#
#                 # img = Image.fromarray(rotated_image.T, 'RGB') # if you want to check the rotations
#                 # img.save('out_rotated.png')
#
#                 print("Predict for rotated image", i, "with", angle, "degrees.")
#                 outputs_rotated = model(data_input_rotated)
#                 outputs_rotated = outputs_rotated[:,:,2:-2,2:-2]
#                 loss_rotated += loss_function(outputs_rotated, data_target_rotated)
#                 correct += score(data_target_rotated, outputs_rotated)
#
#             total_loss = loss + loss_rotated
#             total_loss.backward()
#             optimizer.step()
#
#             #Log
#             loss_value += total_loss.item()
#
#         loss_value = loss_value/((1+n_rotions)*x.shape[0])
#         accuracy = correct/((1+n_rotions)*x.shape[0])
#
#         ''' Validation '''
#
#         model.eval()
#         loss_val_value = 0.0
#         correct_val = 0
#         for i in range(0,val_x.shape[0],BATCH_SIZE):
#             data_val_inputs = val_x[i:BATCH_SIZE+i].to(DEVICE)
#             data_val_targets = val_y[i:BATCH_SIZE+i].to(DEVICE)
#
#             with torch.no_grad():
#
#                 outputs_val = model(data_val_inputs)
#
#                 actual_outputs_val = outputs_val[:,:,2:-2,2:-2]
#                 val_loss = loss_function(actual_outputs_val,data_val_targets)
#              # log
#             loss_val_value +=val_loss.item()
#
#             correct_val += score(data_val_targets,actual_outputs_val)
#
#
#         loss_val_value /= val_x.shape[0]
#         accuracy_val = correct_val/val_x.shape[0]
#
#         val_loss_hist.append(loss_val_value)
#         val_acc_hist.append(accuracy_val)
#         train_loss_hist.append(loss_value)
#         train_acc_hist.append(accuracy)
#
#         if DISPLAY:
#             print(f'Epoch {epoch}, loss: {loss_value:.5f}, accuracy: {accuracy:.3f}, Val_loss: {loss_val_value:.5f}, Val_acc: {accuracy_val:.3f}')
#
#         print(">> Saving Model for Epoch ", str(epoch))
#         checkpoint = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
#         torch.save(checkpoint, './model')
#
#     return val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist


#Plot the logs of the loss and accuracy on the train/validation set
def plot_hist(val_loss_hist, train_loss_hist, val_acc_hist, train_acc_hist):
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
