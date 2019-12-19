# functions were adapted from segment_aerial_images.ipynb provided by the EPFL ML course

import numpy as np
from skimage.util import view_as_windows
import os
import matplotlib.image as mpimg
from constants import *
from scipy import ndimage

""" Extract patches from a given image """
def img_crop(im, w, h):
    list_patches = []
    imgW = im.shape[0]
    imgH = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgH, h):
        for j in range(0, imgW, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


""" Read test and training images """
def read_images(pathFile, training_size, imgName):
    images = []
    for i in range(1, training_size + 1):
        if imgName == 'test_':
            imgPath = pathFile + imgName + '{}.png'.format(i)
        else:
            imgPath = pathFile +  imgName + '{:03d}.png'.format(i)
        if os.path.isfile(imgPath):
            image = mpimg.imread(imgPath)
            images.append(image)
        else:
            print('Image', imgPath, 'can not be found. Check the path.')
    return np.asarray(images)


""" Overlapping patches """
def overlap(image, overlap_amount):
    dim = (PATCH_SIZE, PATCH_SIZE) if len(image.shape) == 2 else (PATCH_SIZE, PATCH_SIZE, 3)
    overlappedWindows = view_as_windows(image, dim, overlap_amount)

    overlappedPatches = []
    for i in range(0, overlappedWindows.shape[0]):
        for j in range(0, overlappedWindows.shape[1]):
            if len(image.shape) == 2:
                overlappedPatches.append(overlappedWindows[i][j])
            else:
                overlappedPatches.append(overlappedWindows[i][j][0])

    return overlappedPatches


""" Read, rotate and patch images """
def read_rotate_patch(images_path, groundTruthPath, training_size, overlap_amount, rotation):
    images = read_images(images_path, training_size, "satImage_")
    labels = read_images(groundTruthPath, training_size, "satImage_")
    imgsPatches = [p for img in images for p in overlap(img, overlap_amount)]
    labelPatches = [p for label in labels for p in overlap(label, overlap_amount)]

    if rotation:
        rotatedImages = imageRotate(images)
        rotatedLabels = imageRotate(labels)

        for img, label in zip(rotatedImages, rotatedLabels):
            patches = zip(overlap(img, overlap_amount), overlap(label, overlap_amount))
            valid_img, valid_label = [], []
            for patch_img, patch_label in patches:
                valid_img.append(patch_img)
                valid_label.append(patch_label)

            imgsPatches.extend(valid_img)
            labelPatches.extend(valid_label)

    return imgsPatches, labelPatches


""" Rotate images using each ROTATION_ANGLES """
def imageRotate(images):
    result = []
    for img in images:
        for angle in ROTATION_ANGLES:  # See constants.py
            rotated = ndimage.rotate(img, angle, reshape=False, axes=(0,1), mode='reflect', order=1)
            result.append(rotated)

    return result
