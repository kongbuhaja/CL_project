import torch, tqdm
from torch.utils.data import DataLoader

from model.utils import *
from data.utils import *
from utils import *
from val import eval

args = args_parse()
args_show(args)
env_set(args.gpus)

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.cpus)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.cpus)

model, start_epoch, best_recall, recalls, save_path = load_model(args.dataset, args.optimizer, args.model, channel=args.channel, 
                                                                   nclasses=len(train_dataset.unique_labels),
                                                                   image_size=args.image_size, load=args.load)
model.to(Device)

epochs = args.epochs
train_iters = len(train_dataloader)

if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr)

scheduler = LR_schedular(optimizer, args.lr_schedular)

lrs = []
for epoch in range(start_epoch, epochs):
    for iter, (_) in enumerate(range(train_iters)):
        lrs += [scheduler.step(epoch*train_iters+iter, epochs*train_iters)]

import matplotlib.pyplot as plt
plt.plot(lrs)
plt.show()
# print(len(val_dataset))