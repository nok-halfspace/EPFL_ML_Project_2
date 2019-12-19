"""
Helpers functions for data visualization
These functions were adapted from segment/tf_aerial_images.ipynb provided by the EPFL ML course
"""
import numpy as np

def img_float_to_uint8(img):
    """
    Transform float image to uint image.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

""" Concatenate an image and its groundtruth. """
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = gray_to_rgb(gt_img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

""" Overlay prediction on the aerial image. """
def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

""" Creates a RGB image from a gray image using the same values for all the 3 channels """
def gray_to_rgb(grayImage):
    intValuesImage = img_float_to_uint8(grayImage)
    heigth  = grayImage.shape[1]
    width   = grayImage.shape[0]
    rgbImage = np.zeros((width, heigth, 3), dtype=np.uint8)
    rgbImage[:,:,0] = intValuesImage
    rgbImage[:,:,1] = intValuesImage
    rgbImage[:,:,2] = intValuesImage
    return rgbImage
