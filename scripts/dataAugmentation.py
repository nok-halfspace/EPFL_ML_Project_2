from PIL import Image
from imgaug import augmenters as iaa
import os
import imageio
from constants import *


# For now : 3 rotations for each image : 45, 90, 135 degrees + padding reflect to get image of 400*400 

def augment_dataset(TRAINING_SIZE, data_dir, path, rotate = True, save = True):
    train_data_filename = data_dir + path
    num_images = TRAINING_SIZE
    
    # Defining the transformations : 
    rotate45 = iaa.Affine(rotate=45, mode = 'reflect')
    rotate90 = iaa.Affine(rotate=90, mode = 'reflect')
    rotate135 = iaa.Affine(rotate=135, mode = 'reflect')

    for i in range(1, num_images+1):

        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        print(image_filename)
        if os.path.isfile(image_filename):
            img = imageio.imread(image_filename)
           
            if (rotate == True):
                r_img45 = rotate45.augment_images([img])[0]
                r_img90 = rotate90.augment_images([img])[0]
                r_img135 = rotate135.augment_images([img])[0]
                
            if (save == True):
                Image.fromarray(r_img45).save("../Datasets/Rotations/" + path + imageid + "_45.png")
                Image.fromarray(r_img90).save("../Datasets//Rotations/" + path  + imageid + "_90.png")
                Image.fromarray(r_img135).save("../Datasets//Rotations/" + path + imageid + "_135.png")
                     
                        


data_dir = '../Datasets/training/'

train_data_filename = 'images/'
train_labels_filename = 'groundtruth/'

augment_dataset(TRAINING_SIZE, data_dir, train_data_filename)
augment_dataset(TRAINING_SIZE, data_dir, train_labels_filename)
