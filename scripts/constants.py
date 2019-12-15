import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DISPLAY = False
TRAINING_SIZE = 100 # Debug purposes
NUM_EPOCHS = 10
NR_TEST_IMAGES = 50
rotateFlag = False
RATIO = 0.9
IMG_PATCH_SIZE = 16
BATCH_SIZE = 1

submissionFileName = "latestSubmission.csv"
data_dir = '../Datasets/training/'
test_dir = '../Datasets/test_set_images/test_'
train_data_filename = 'images/'
train_labels_filename = 'groundtruth/'
MODEL_PATH = "./model.pkg"

RELOAD_MODEL = False
DISPLAY = False
