import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DISPLAY = False
TOTAL_TRAINING_SIZE = 100
TRAINING_SIZE = RATIO * TOTAL_TRAINING_SIZE # Debug purposes
VAL_SIZE = (1 - RATIO) * TOTAL_TRAINING_SIZE# debug purpose
NUM_EPOCHS = 1
NR_TEST_IMAGES = 50
ROTATION = True
ROTATION_ANGLES = [45, 90, 135, 180, 225, 270, 315]
RATIO = 0.9
IMG_PATCH_SIZE = 16
NUM_CLASSES = 1

SUBMISSION_PATH           = '../submissions/'
TRAIN_IMAGE_PATH          = '../Datasets/training/images/'
TRAIN_GROUNDTRUTH_PATH    = '../Datasets/training/groundtruth/'
TEST_IMAGE_PATH           = '../Datasets/test_set_images/'
PREDICTED_PATH            = '../Datasets/predictions/'

BATCH_SIZE = 128

# TODO: Change these names
PATCH_SIZE      = 80
OVERLAP         = True # Always true : should remove that 
OVERLAP_AMOUNT  = 20
TEST_IMG_SIZE   = 608

RELOAD_MODEL = False
DISPLAY = False
