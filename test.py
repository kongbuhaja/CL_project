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

def setup(rank, world_size, model):
    print(rank, world_size)
    print(1)
    dist.init_process_group(backend='nccl', 
                            init_method=f'tcp://166.104.30.82:4458', 
                            rank=rank,
                            world_size=world_size)
    print(2)
    torch.cuda.set_device(rank)
    print(3)
    model = model.to(rank)
    print(4)
    model = DDP(model, device_ids=[rank])
    print(5)
    return model

# Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Devices = [torch.device(f'cuda:{i}') for i in range(len(args.gpus.split(',')))] if torch.cuda.is_available() else 'cpu'
# Device = torch.device("cpu")

def process(rank, model_type, world_size, args):
    torch.manual_seed(42)
    
    dist.init_process_group(backend='nccl',
                            init_method='tcp://localhost:4458',
                            rank=rank,
                            world_size=world_size)
    
    train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model, start_epoch, best_recall, recalls = load_model(args.dataset, model_type, channel=args.channel, 
                                                        nclasses=len(train_dataset.unique_labels),
                                                        image_size=args.image_size, load=args.load)
    model = DDP(model.to(rank), device_ids=[rank])

    dist.destroy_process_group()


if __name__ == '__main__':
    args = arg_parse()
    env_set(args.gpus)
    gpus = [g for g in range(len(args.gpus.split(',')))]
    model_types = args.model.split(',')

    processes = []
    world_size = len(gpus)
    for gpu, model_type in zip(gpus, model_types):
        p = Process(target=process, args=(gpu, model_type, world_size, args))
        p.start()
        processes += [p]

    for p in processes:
        p.join()

