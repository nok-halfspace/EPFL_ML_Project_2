import torch

# To be adjusted based on the images we have
N = 10
BATCH_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Whether to display the logs during training
DISPLAY = False
TRAINING_SIZE = 4 # Debug purposes
NUM_EPOCHS = 1
N_CLASSES = 2
NR_TEST_IMAGES = 2 # 50
rotateFlag = False
RATIO=0.5

submissionFileName = "latestSubmission.csv"
data_dir = '../Datasets/training/'
test_dir = '../Datasets/test_set_images/test_'
train_data_filename = 'images/'
train_labels_filename = 'groundtruth/'
RELOAD_MODEL = True
MODEL_PATH = "./model.pkg"
