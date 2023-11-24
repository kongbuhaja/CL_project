import torch
import cv2
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader

from model.utils import *
from data.utils import *
from utils import *
from val import eval

import numpy as np
import torchsummary

np.random.seed(42)
torch.manual_seed(42)

args = arg_parse()
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Device = torch.device("cpu")

train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

model, start_epoch, best_recall, recalls = load_model(args.dataset, args.model, channel=args.channel, 
                                                      nclasses=len(train_dataset.unique_labels),
                                                      image_size=args.image_size, load=args.load)
model.to(Device)
# save_recall(args.dataset, args.model, recalls)

torchsummary.summary(model, tuple(args.image_size + [3]))