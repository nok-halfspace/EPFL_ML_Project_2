import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import models
import torch.optim as optim
from torch.autograd import Variable

TRAINING_SIZE = 20
NUM_EPOCHS = 100

# This function returns a list of patches from image (3D),
# each patch has a size of patch_h * patch_w
def getPatches(image, patch_h, patch_w):
    patches = []
    width = image.shape[1]
    height = image.shape[2]
    # print(width, height)
    for i in range(0, height, patch_h):
        for j in range(0, width, patch_w):
            patch = image[:, j:j+patch_w, i:i+patch_h]
            patches.append(patch)
    return patches

def extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename):

    to_tensor = transforms.ToTensor() #ToTensor transforms the image to a tensor with range [0,1]
    num_images = TRAINING_SIZE
    imgs = []

    for i in range(1, num_images+1):

        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        print(image_filename)
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)

            t_img = to_tensor(img).unsqueeze(0) # 1 x 3 [rgb] x 400 x 400

            imgs.append(t_img)
    return imgs # length TRAINING_SIZE

def main():

    data_dir = '../Datasets/training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'
    patch_h = 32
    patch_w = 32


    imgs =  extract_feature_vectors(TRAINING_SIZE, data_dir, train_data_filename)
    labels = extract_feature_vectors(TRAINING_SIZE, data_dir, train_labels_filename)

    input = torch.Tensor(imgs[:][0])

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss = torch.nn.BCELoss()
    model = models.UNET().to(DEVICE)
    optimize = optim.Adam(model.parameters())

    output = loss(model(input), labels)
    output.backward()

    print(output)


if __name__== "__main__":
    main()
