"""
Image visualization / transformation helpers.
"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy
numpy.seterr(divide='ignore', invalid='ignore')

IMG_PATCH_SIZE = 16

def probability_to_prediction(outputs):
    predictions = (outputs > 0.5).squeeze()
    predictions = predictions.cpu().numpy().astype(int)
    return predictions


def generate_predictions(testing_size, test_image_size, test_patch_size, labels, path):
    """
    Generate prediction image from labels.
    """
    STEP = int(len(labels) / testing_size)
    assert STEP == (test_image_size**2 / test_patch_size**2)


    for i in range(0, len(labels), STEP):
        labels_for_patch = labels[i: i+STEP]
        prediction = label_to_img(test_image_size, test_image_size, test_patch_size, test_patch_size, labels_for_patch)

        k = int((i+STEP)/STEP)
        img = Image.fromarray(to_rgb(prediction))
        img.save(path + "satImage_%.3d" % k + ".png")

# from tf_aerial_images
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            array_labels[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return array_labels


def extract_patches(images, patch_size):
    """
    Extract all patches from the images.
    """
    patches = []
    for im in images:
        patches.extend(img_crop(im, patch_size, patch_size))

    return patches

def img_float_to_uint8(img):
    """
    Transform float image to uint image.
    """
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * 255).round().astype(numpy.uint8)
    return rimg

def to_rgb(gt_img):
    """
    Get RGB image from groundtruth.
    """
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(gt_img)
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c

def make_img_overlay(img, predicted_img):
    """
    Display prediction overlay on top of original image.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def concatenate_images(img, gt_img):
    """
    Concatenate two images side by side.
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


# Assign a label to a patch v
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
