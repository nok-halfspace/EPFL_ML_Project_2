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

        image = test_imgs[i].to(DEVICE)
        image = torch.unsqueeze(image, 0)
        image = network(image)
        image = image[0]

        image = torch.argmax(softMax(image), 0)
        image = image.cpu().numpy()
        image = image[2:610, 2:610]
        print(image.shape)
        Image.fromarray(255*image.astype('uint8')).save(filename)

    return filenames_list
