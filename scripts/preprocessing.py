
import torchvision.transforms as transforms
import torch
from constants import *
import os
from PIL import Image

def readTrainingImages(TRAINING_SIZE, data_dir, path, rotate = False, save = False):
    train_data_filename = data_dir + path
    to_tensor = transforms.ToTensor() #ToTensor transforms the image to a tensor with range [0,1]
    num_images = TRAINING_SIZE
    imgs = []
    r_imgs = []

    for i in range(1, num_images+1):

        imageid = "satImage_%.3d" % i
        image_filename = train_data_filename + imageid + ".png"
        print(image_filename)
        if os.path.isfile(image_filename):
            img = Image.open(image_filename)
            t_img = to_tensor(img) #3 [rgb] x 400 x 400
            imgs.append(t_img)

            if (rotate == True):
                width, height = img.size
                # Rotate images

                for degree in range(360):
                    rotated_img = img.rotate(degree)
                    # Crop to remove black parts
                    left     = width / 4 * 0.58
                    top      = height / 4 * 0.58
                    right    = width - left
                    bottom   = height - top
                    rotated_img = rotated_img.crop((left, top, right, bottom))
                    if (save == True):
                        # optional, save new image on disk, to see the effect
                        rotated_img.save("../Rotations/"+path + imageid + "_"+str(degree)+".png")
                    rt_img = to_tensor(rotated_img) # 3 [rgb] x 284 x 284
                    r_imgs.append(rt_img)
        else:
            print(image_filename, "is not a file, follow the README instruction to run the project. (check path)", file=sys.stderr)
            sys.exit()

    if rotate:
        r_imgs = torch.stack(r_imgs)
    imgs = torch.stack(imgs)

    return imgs, r_imgs

def readTestImages(test_directory, num_images):
    imgs = []
    to_tensor = transforms.ToTensor()
    current_image_path = ""
    for i in range(1, num_images+1):
        current_image_path = test_directory + str(i) + "/test_" + str(i) + ".png"
        if os.path.isfile(current_image_path):
            img = Image.open(current_image_path)
            t_img = to_tensor(img) # 3 [rgb] x 600 x 600
            imgs.append(t_img)

        else:
            print(current_image_path, "is not a file, follow the README instruction to run the project. (check path)", file=sys.stderr)
            sys.exit()

    imgs = torch.stack(imgs)
    return imgs
    # Assign a one-hot label to each pixel of a ground_truth image
    # can be improved using scatter probably
    # or see how it is done in the tf_aerial.py

def value_to_label_by_patches(img):

    patch_size = 16

    img = img.squeeze().numpy()
    x_patch = int(img.shape[0]/patch_size)
    y_patch = int(img.shape[1]/patch_size)
    labels = torch.randn((x_patch*patch_size,y_patch*patch_size)).to(DEVICE)

    foreground_threshold = 0.5

    for x in range(x_patch) :
        for y in range(y_patch) :

            # find average value over the given subset of the image
            val = img[x*patch_size:x*patch_size+patch_size, y*patch_size:y*patch_size+patch_size]
            val = val.mean()

            if val > foreground_threshold:  # road
                labels[x*patch_size:x*patch_size+patch_size,y*patch_size:y*patch_size+patch_size] = torch.tensor(1.0)
            else:  # bgrd
                labels[x*patch_size:x*patch_size+patch_size,y*patch_size:y*patch_size+patch_size] = torch.tensor(0.0)
    return labels.long()
