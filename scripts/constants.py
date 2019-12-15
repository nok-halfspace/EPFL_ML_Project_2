import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DISPLAY = False
TRAINING_SIZE = 2 # Debug purposes
NUM_EPOCHS = 1
NR_TEST_IMAGES = 50
rotateFlag = False
RATIO = 0.75
IMG_PATCH_SIZE = 16
DROPOUT = 0.2
NUM_CLASSES = 1

SUBMISSION_PATH, CHECKPOINT_PATH = '../submissions/', '../checkpoints/'
TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH = '../Datasets/training/images/', '../Datasets/training/groundtruth/'
TEST_IMAGE_PATH, TEST_LABEL_PATH = '../Datasets/test_set_images/', '../Datasets/test/predictions/'

BATCH_SIZE = 128

# TODO: Change these names
PATCH_SIZE, OVERLAP, OVERLAP_AMOUNT = 80, True, 20
TEST_IMG_SIZE, TEST_PATCH_SIZE = 608, 16

submissionFileName = "latestSubmission.csv"
data_dir = '../Datasets/training/'
test_dir = '../Datasets/test_set_images/test_'
train_data_filename = 'images/'
train_labels_filename = 'groundtruth/'
MODEL_PATH = "./model.pkg"

RELOAD_MODEL = False
DISPLAY = False
