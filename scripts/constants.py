import torch

# To be adjusted based on the images we have 
N = 10
BATCH_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Whether to display the logs during training
DISPLAY = False
TRAINING_SIZE = 2 # Debug purposes 
NUM_EPOCHS = 1
N_CLASSES = 2