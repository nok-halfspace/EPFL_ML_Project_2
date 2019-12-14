import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

submissionFileName = "latestSubmission.csv"
data_dir = '../Datasets/training/'
test_dir = '../Datasets/test_set_images/test_'
train_data_filename = 'images/'
train_labels_filename = 'groundtruth/'
MODEL_PATH = "./model.pkg"

RELOAD_MODEL = False
DISPLAY = False
rotateFlag = False

BATCH_SIZE = 1
TRAINING_SIZE = 4
NUM_EPOCHS = 3
N_CLASSES = 2
NR_TEST_IMAGES = 2 
RATIO=0.5
IMG_PATCH_SIZE = 16

