import torch, cv2, os
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from model.utils import *
from data.utils import *
from utils import *
from val import eval

import numpy as np

args = args_parse()
args_show(args)
env_set(args.gpus)

train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

print(len(train_dataloader))