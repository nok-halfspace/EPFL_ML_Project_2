import torch
import torchvision.transforms as transforms

from PIL import Image
from training import *
from mask_to_submission import *
from submission_to_mask import *

def test_and_save_predictions(network, test_imgs):
    print(test_imgs.shape)
    softMax = torch.nn.Softmax(0)
    filenames_list = []
    for i in range(0, NR_TEST_IMAGES): # test_imgs.shape[0]
        print("Create prediction for image #", i+1)
        filename = "../Datasets/Prediction/prediction_" + str(i+1) + ".png"
        filenames_list.append(filename)

        image = test_imgs[i]
        image = torch.unsqueeze(image, 0)
        image = network(image)
        image = image[0]

        image = torch.argmax(softMax(image), 0)
        image = image.numpy()
        image = image[2:610, 2:610]
        print(image.shape)
        Image.fromarray(255*image.astype('uint8')).save(filename)

    return filenames_list

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
