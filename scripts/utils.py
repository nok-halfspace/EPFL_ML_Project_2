import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy
from constants import *
from prepareData import img_crop
from imageFunctions import gray_to_rgb
numpy.seterr(divide='ignore', invalid='ignore')

""" Get 1 or 0 values for predictions """
def probability_to_prediction(outputs):
    predictions = (outputs > 0.5).squeeze()
    predictions = predictions.cpu().numpy().astype(int)
    return predictions


""" Get patches from images """
def getPatches(images):
    patches = []
    for im in images:
        patches.extend(img_crop(im, IMG_PATCH_SIZE, IMG_PATCH_SIZE))

    return patches


""" Create image from labels """ # from tf_aerial_images
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            array_labels[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return array_labels


""" Assign a label to a patch v """
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.mean(v)
    if df > foreground_threshold:  # road
        return 1
    else:  # bgrd
        return 0


def patch_prediction(imgs, img_size, patch_size):
    imgs_predicted = []
    for i in range(len(imgs)):
        img = imgs[i]
        img_patched = numpy.zeros([img_size, img_size])
        for i in range(0, img_size, patch_size):
            for j in range(0, img_size,patch_size):
                patch = img[i : i+patch_size, j : j+patch_size]
                img_patched[i : i+patch_size, j : j+patch_size] = value_to_class(patch)
        imgs_predicted.append(img_patched)
    return imgs_predicted



def reconstruct_img(testing_size, test_image_size, test_patch_size, labels, path):
    """
    Reconstruct image based on the patches predictions
    """
    # Number of patches per image
    patch_img = int(len(labels) / testing_size)

    for i in range(0, len(labels), patch_img):
        labels_img = labels[i: i+patch_img]
        array_labels = label_to_img(test_image_size, test_image_size, test_patch_size, test_patch_size, labels_img)
        img = Image.fromarray(gray_to_rgb(array_labels))
        idx_img = 1 + int(i/patch_img)
        img.save(path + "satImage_%.3d" % idx_img + ".png")
