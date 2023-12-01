import torch
import cv2
# from torchvision.transforms import Lambda
from torch.utils.data import DataLoader

from model.utils import *
from data.utils import *
from utils import *
from val import eval

import numpy as np
import torchsummary

r = 13
np.random.seed(r)
torch.manual_seed(r)

args = arg_parse()
# Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Device = torch.device("cpu")

train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
# print(len(val_dataset))

# model, start_epoch, best_recall, recalls = load_model(args.dataset, args.model, channel=args.channel, 
#                                                       nclasses=len(train_dataset.unique_labels),
#                                                       image_size=args.image_size, load=args.load)
# # model.qq()
# model.to(Device)

x, y = next(iter(train_dataloader))
x_, y_ = next(iter(val_dataloader))
for xx, xx_ in zip(x, x_):
    img = np.concatenate([(xx.numpy()*255).astype(np.uint8), (xx_.numpy()*255).astype(np.uint8)], 1)
    cv2.imshow('t', img)
    if cv2.waitKey()==27:
        break
cv2.destroyAllWindows()
# save_recall(args.dataset, args.model, recalls)

# torchsummary.summary(model, tuple(args.image_size + [3]))