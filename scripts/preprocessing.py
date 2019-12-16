import numpy as np
from imgaug import augmenters as iaa
from skimage import img_as_ubyte, img_as_float
from skimage.util import view_as_windows
import os
import matplotlib.image as mpimg

def read_images(pathFile, training_size, imgName):
    print ('Reading', training_size, 'images...')
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


"""
Patch extraction helpers.
"""
def patchify_overlap(im, patch_len, overlap_amount):
    """
    Extract overlapping patches from image.
    """
    is_2D = len(im.shape) == 2

    if is_2D:
        window_shape = (patch_len, patch_len)
    else:
        window_shape = (patch_len, patch_len, 3)

    overlapped_windows = view_as_windows(im, window_shape, overlap_amount)

    patches = []
    for i in range(0, overlapped_windows.shape[0], 1):
        for j in range(0, overlapped_windows.shape[1], 1):
            if is_2D:
                patches.append(overlapped_windows[i][j])
            else:
                patches.append(overlapped_windows[i][j][0])

    return patches

def patchify(im, patch_len):
    """
    Extract non-overlapping patches from image.
    If patch_len is not multiple of the image size, image is mirrored.
    """
    patches = []
    width, height = im.shape[0], im.shape[1]
    is_2D = len(im.shape) == 2

    # Mirror image for a length of one patch to make sure we get the whole image in patches
    im = mirror(im, patch_len)

    # Build the right number of patches out of the mirrored image
    for i in range(0, width, patch_len):
        for j in range(0, height, patch_len):
            if is_2D:
                patch = im[j:j+patch_len, i:i+patch_len]
            else:
                patch = im[j:j+patch_len, i:i+patch_len, :]

            patches.append(patch)
    return patches


def mirror(im, length):
    """
    Mirror an image on the right on length pixels
    """
    width, height = im.shape[0], im.shape[1]
    is_2D = len(im.shape) == 2

    if is_2D:
        right_flipped = np.fliplr(im[:, width - length:])
    else:
        right_flipped = np.fliplr(im[:, width - length:, :])

    right_mirrored = np.concatenate((im, right_flipped), axis=1)

    if is_2D:
        bottom_flipped = np.flipud(right_mirrored[height - length:, :])
    else:
        bottom_flipped = np.flipud(right_mirrored[height - length:, :, :])

    mirrored = np.concatenate((right_mirrored, bottom_flipped), axis=0)
    return mirrored


def read_rotate_patch(images_path, labels_path, training_size, patch_size, overlap, overlap_amount, rotation, rotation_angles):
    images = read_images(images_path, training_size, "satImage_")
    labels = read_images(labels_path, training_size, "satImage_")

    # Get patches
    if overlap:
        image_patches = [patch for im in images for patch in patchify_overlap(im, patch_size, overlap_amount)]
        label_patches = [patch for label in labels for patch in patchify_overlap(label, patch_size, overlap_amount)]
    else:
        image_patches = [patch for im in images for patch in patchify(im, patch_size)]
        label_patches = [patch for label in labels for patch in patchify(label, patch_size)]

    patches = zip(image_patches, label_patches)

    # Rotation needs to be applied on whole image
    if rotation:
        images_rot = imageRotate(images, rotation_angles)
        labels_rot = imageRotate(labels, rotation_angles)

        for im, label in zip(images_rot, labels_rot):
            p = patchify_no_corner(im, label, patch_size, overlap, overlap_amount)
            image_patches.extend(p[0])
            label_patches.extend(p[1])

    return image_patches, label_patches


def imageRotate(images, angles):
    result = []
    for im in images:
        for angle in angles:
            rotated = iaa.Affine(rotate=angle).augment_image(im)
            result.append(rotated)

    return result

def patchify_no_corner(img, label, patch_size, overlap, overlap_amount):
    """
    Patchify and remove invalid corners due to rotation for both image and label.
    """
    if overlap:
        patches = zip(patchify_overlap(img, patch_size, overlap_amount), patchify_overlap(label, patch_size, overlap_amount))
    else:
        patches = zip(patchify(img, patch_size), patchify(label, patch_size))

    valid_img, valid_label = [], []

    for patch_img, patch_label in patches:
        if not is_corner(patch_img):
            valid_img.append(patch_img)
            valid_label.append(patch_label)

    return valid_img, valid_label

def is_corner(patch):
    """
    True of the given patch is likely to be in a rotation corner.
    """
    lt = not np.any(patch[0][0])
    rt = not np.any(patch[0][-1])
    lb = not np.any(patch[-1][0])
    rb = not np.any(patch[-1][-1])

    return lt or rt or lb or rb
