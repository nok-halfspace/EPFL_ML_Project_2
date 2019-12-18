import torch
import math

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TOTAL_TRAINING_SIZE = 100
RATIO               = 0.9
TRAINING_SIZE       = int(RATIO * TOTAL_TRAINING_SIZE) # Debug purposes
VAL_SIZE            = math.ceil((1 - RATIO) * TOTAL_TRAINING_SIZE) # debug purpose
NUM_EPOCHS          = 10
NR_TEST_IMAGES      = 50
ROTATION            = True
ROTATION_ANGLES     = [30, 45, 60, 90, 135]
IMG_PATCH_SIZE      = 16
BATCH_SIZE          = 128
PATCH_SIZE          = 80
OVERLAY_SIZE        = 20
TEST_IMG_SIZE       = 608
DROPOUT             = 0.0

SUBMISSION_PATH           = '../submissions/'
TRAIN_IMAGE_PATH          = '../Datasets/training/images/'
TRAIN_GROUNDTRUTH_PATH    = '../Datasets/training/groundtruth/'
TEST_IMAGE_PATH           = '../Datasets/test_set_images/'
PREDICTED_PATH            = '../Datasets/predictions/'
MODEL_PATH                = '../model.weights'
