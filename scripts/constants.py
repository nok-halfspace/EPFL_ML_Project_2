import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DISPLAY = False
TRAINING_SIZE = 10 # Debug purposes
NUM_EPOCHS = 1
NR_TEST_IMAGES = 5
ROTATION = True
ROTATION_ANGLES = [45, 90, 135, 180, 225, 270, 315]
RATIO = 0.9
IMG_PATCH_SIZE = 16
DROPOUT = 0.2
NUM_CLASSES = 1

SUBMISSION_PATH           = '../submissions/',
CHECKPOINT_PATH           = '../checkpoints/'
TRAIN_IMAGE_PATH          = '../Datasets/training/images/'
TRAIN_GROUNDTRUTH_PATH    = '../Datasets/training/groundtruth/'
TEST_IMAGE_PATH           = '../Datasets/test_set_images/'
PREDICTED_PATH            = '../Datasets/predictions/'

BATCH_SIZE = 128

# TODO: Change these names
PATCH_SIZE      = 80
OVERLAP         = True
OVERLAP_AMOUNT  = 20
TEST_IMG_SIZE   = 608

submissionFileName = "latestSubmission.csv"
data_dir = '../Datasets/training/'
test_dir = '../Datasets/test_set_images/test_'
train_data_filename = 'images/'
train_labels_filename = 'groundtruth/'
MODEL_PATH = "./model.pkg"

RELOAD_MODEL = False
DISPLAY = False
