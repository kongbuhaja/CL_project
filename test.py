import torch
import cv2
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader

from model.utils import load_model, save_model
from data.utils import *
from utils import arg_parse, loss_function, LR_schedular

import torchsummary
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

args = arg_parse()
# # Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Device = torch.device("cpu")

# train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

model, start_epochs, best_loss, ap = load_model(args.dataset, args.model, 10, 32, load=args.load)
# torchsummary.summary(model, (100,100,3), device='cpu')


