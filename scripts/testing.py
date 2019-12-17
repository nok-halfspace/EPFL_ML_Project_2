import torch
import torchvision.transforms as transforms

from PIL import Image
from training import *
from mask_to_submission import *
from submission_to_mask import *
import numpy as np
from torch.autograd import Variable

def predict_test_images(model, loader):
    ouputPredicted = []
    model.eval()

    for data in loader:
        patches = data.to(DEVICE)

        with torch.no_grad():
            outputs = model(patches)
            predicted = np.rint(outputs.squeeze().data.cpu().numpy())
            ouputPredicted.append(predicted)

    return ouputPredicted
