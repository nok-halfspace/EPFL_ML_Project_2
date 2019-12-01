import torch

# To be adjusted based on the images we have 
N = 10
BATCH_SIZE = 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Whether to display the logs during training
DISPLAY = False
