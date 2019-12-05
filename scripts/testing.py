import torch

from training import *

def test(network, score, x, y):
    correct = 0
    for i in range(0,x.shape[0],BATCH_SIZE):
        data_inputs = x[i:BATCH_SIZE+i]
        data_targets = y[i:BATCH_SIZE+i]
        
        outputs = network(data_inputs)        
        correct += score(data_targets,outputs) 

    accuracy = correct/x.shape[0]
    return accuracy
    
def mean_std(x_ls):
    x_mean = torch.mean(torch.tensor(x_ls)).item()
    x_std = torch.std(torch.tensor(x_ls)).item()
    return x_mean, x_std


# Run the round times the whole process to see the variability with different initialisations (=> may not be feasible in our case)
def round_test(create_net, x, y, epoch, score, x_test, y_test):
    train_ls = []
    test_ls = []
    true_test_ls = []
    
    for i in range(ROUNDS):
        torch.manual_seed(2*i+1)
        network, loss_function, optimizer = create_net()
        
        print("Round {}".format(i),end="\r")
        val_loss_hist,train_loss_hist,val_acc_hist,train_acc_hist = training(network, loss_function, optimizer, score, x, y, epoch,val_split=0.01)
        
        #log
        train_ls.append(train_acc_hist[-1])
        test_ls.append(test(network,score,x_test,y_test))
            
        #Clear cache of GPU
        del network
        if torch.cuda.is_available():
            torch._C._cuda_emptyCache()
        
    return mean_std(train_ls),mean_std(test_ls),mean_std(true_test_ls)
    