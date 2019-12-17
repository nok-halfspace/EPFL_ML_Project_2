import torch
import torchvision.transforms as transforms

from PIL import Image
from training import *
from mask_to_submission import *
from submission_to_mask import *
import numpy as np
from torch.autograd import Variable
from helpers import probability_to_prediction

def predict_test_images(model, loader):
    outputPredicted = []
    model.eval()

    for data in loader:
        imgs = data.to(DEVICE)

        with torch.no_grad():
            outputs = model(imgs)
            predictions = probability_to_prediction(outputs)
            outputPredicted.append(predictions)

    return outputPredicted
