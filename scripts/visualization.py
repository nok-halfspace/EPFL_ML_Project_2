"""
Helpers functions for data visualization
These functions were taken and adapted from segment_aerial_images.ipynb provided by the EPFL ML course
"""
import numpy

def img_float_to_uint8(img):
    """
    Transform float image to uint image.
    """
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * 255).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    """
    Concatenate an image and its groundtruth.
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = gray_to_rgb(gt_img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    """
    Overlay prediction on the aerial image.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def gray_to_rgb(gt_img):
    """
    TO DO 
    """ 
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    gt_img8 = img_float_to_uint8(gt_img)
    gt_img_3c[:,:,0] = gt_img8
    gt_img_3c[:,:,1] = gt_img8
    gt_img_3c[:,:,2] = gt_img8
    return gt_img_3c
    
