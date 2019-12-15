
"""
Channel augmentation main function.
"""
import numpy as np
from imgaug import augmenters as iaa
from skimage import img_as_ubyte, img_as_float
from skimage.util import view_as_windows
import os
import matplotlib.image as mpimg
"""
Image augmentation configuration wrapper.
"""
class ImageAugmentationConfig:
    """
    Encapsulate an augmentation configuration.
    Example:

        config = ImageAugmentationConfig()
        config.rotation([20, 80])
        config.flip()
        config.edge()
        config.blur()
        config.contrast()
    """
    def __init__(self):
        self.do_rotation = False
        self.do_flip = False
        self.augment_channels = False
        self.do_edge = False
        self.do_blur = False
        self.do_contrast = False
        self.do_convolve = False
        self.do_invert = False

    def rotation(self, angles):
        """
        Add rotations.
        """
        self.do_rotation = True
        self.rotation_angles = angles

    def flip(self):
        """
        Add flip transformation.
        """
        self.do_flip = True

    def edge(self):
        """
        Add edge augmentation.
        """
        self.augment_channels = True
        self.do_edge = True

    def contrast(self):
        """
        Add contrast augmentation.
        """
        self.augment_channels = True
        self.do_contrast = True

    def convolve(self):
        """
        Add convolution augmentation.
        """
        self.augment_channels = True
        self.do_convolve = True

    def invert(self):
        """
        Add invertion augmentation.
        """
        self.augment_channels = True
        self.do_invert = True

    def blur(self, sigma=2):
        """
        Add blur augmentation.
        """
        self.augment_channels = True
        self.do_blur = True
        self.blur_sigma = sigma



def augment_channels(images, aug_config):
    """
    Augment each image in images with the channel transformation given in aug_config.
    """
    augmented_images = []

    # Instantiate transformations
    if aug_config.do_blur:
        blur = iaa.GaussianBlur(sigma=aug_config.blur_sigma)
    if aug_config.do_edge:
        edge = iaa.EdgeDetect(alpha=1)
    if aug_config.do_contrast:
        contrast = iaa.ContrastNormalization((0.5, 1.5))
    if aug_config.do_convolve:
        convolve = iaa.Convolve(matrix=np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]]))
    if aug_config.do_invert:
        invert = iaa.Invert(1)

    # Augment each image
    for im in images:
        augmented = im
        if aug_config.do_blur:
            aug = img_as_float(blur.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        if aug_config.do_edge:
            aug = img_as_float(edge.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        if aug_config.do_contrast:
            aug = img_as_float(contrast.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        if aug_config.do_convolve:
            aug = img_as_float(convolve.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        if aug_config.do_invert:
            aug = img_as_float(invert.augment_image(img_as_ubyte(im)))
            augmented = np.dstack((augmented, aug))

        augmented_images.append(augmented)

    return augmented_images


"""
Images loading helpers.
"""
def extract_images(filepath, indices, imgName):
    """
    Load images with given indices from the filepath.
    Note that the classical file name convention of the project is used.
    """
    imgs = []
    print ('Loading ', len(indices),' test images...')

    # Load all images
    for i in indices:
        if imgName == 'test_':
            filename = filepath + imgName + '{}.png'.format(i)
        else:
            filename = filepath +  imgName + '{:03d}.png'.format(i)
        if os.path.isfile(filename):
            img = mpimg.imread(filename)
            imgs.append(img)
        else:
            print('File {} does not exists'.format(filename))

    print('done')
    return np.asarray(imgs)


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



"""
Main functions responsible for loading and transforming images.
"""
def prepare_train_patches(images_path, labels_path, indices, patch_size, overlap, overlap_amount, aug_config):
    """
    Load, patchify and augment images and labels for training.
    """

    # Load images and labels
    images = extract_images(images_path, indices, "satImage_")
    labels = extract_images(labels_path, indices, "satImage_")

    # Get patches
    if overlap:
        image_patches = [patch for im in images for patch in patchify_overlap(im, patch_size, overlap_amount)]
        label_patches = [patch for label in labels for patch in patchify_overlap(label, patch_size, overlap_amount)]
    else:
        image_patches = [patch for im in images for patch in patchify(im, patch_size)]
        label_patches = [patch for label in labels for patch in patchify(label, patch_size)]

    if not aug_config:
        return image_patches, label_patches

    patches = zip(image_patches, label_patches)

    # Rotation needs to be applied on whole image
    if aug_config.do_rotation:
        images_rot = rotate_images(images, aug_config.rotation_angles)
        labels_rot = rotate_images(labels, aug_config.rotation_angles)

        for im, label in zip(images_rot, labels_rot):
            p = patchify_no_corner(im, label, patch_size, overlap, overlap_amount)
            image_patches.extend(p[0])
            label_patches.extend(p[1])

    # Flip each patch horizontally
    images_flipped = []
    labels_flipped = []
    if aug_config.do_flip:
        flip_hor = iaa.Fliplr(0.5).to_deterministic()
        flip_ver = iaa.Flipud(0.5).to_deterministic()
        images_flipped.extend(flip_hor.augment_images(image_patches))
        images_flipped.extend(flip_ver.augment_images(image_patches))
        labels_flipped.extend(flip_hor.augment_images(label_patches))
        labels_flipped.extend(flip_ver.augment_images(label_patches))

    image_patches.extend([im.copy() for im in images_flipped])
    label_patches.extend([im.copy() for im in labels_flipped])

    # For all the patches (even new ones), augment channels
    if aug_config.augment_channels:
        image_patches = augment_channels(image_patches, aug_config)

    return image_patches, label_patches


"""
Image rotation related helpers.
"""
def rotate_images(images, angles):
    """
    Rotates all the images by all the given angles.
    """
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
