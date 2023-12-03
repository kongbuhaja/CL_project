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

model, start_epoch, best_recall, recalls = load_model(args.dataset, args.model, channel=args.channel, 
                                                      nclasses=len(train_dataset.unique_labels),
                                                      image_size=args.image_size, load=args.load)
model.to(Device)

epochs = args.epochs
train_iters = len(train_dataloader)

optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)
scheduler = LR_schedular(optimizer, args.lr_schedular)
loss_fn = loss_function(args.loss, len(train_dataset.unique_labels))

for epoch in range(start_epoch+1, epochs+1):
    # train
    train_loss = 0.
    model.train()
    train_tqdm = tqdm.tqdm(train_dataloader, total=train_iters, ncols=120, desc=f'Train epoch {epoch}/{epochs}', ascii=' =', colour='red')
    for iter, (x_data, y_data) in enumerate(train_tqdm):
        pred = model(x_data.to(Device))
        loss = loss_fn(pred, y_data[..., 0].to(Device))
        loss.backward()
        scheduler.step(epoch*train_iters+iter, epochs*train_iters)

        train_loss += loss.item()
        train_tqdm.set_postfix_str(f'| iter: {iter+1}/{train_iters} | lr: {scheduler.lr:.4f}, total_loss: {train_loss/(iter+1):.4f}')

    # eval
    if epoch%args.eval_term==0:
        recalls += [eval(model, val_dataloader, loss_fn, Device)]
        if best_recall < recalls[-1]:
            best_recall = recalls[-1]
            save_model(model, args.dataset, args.model, epoch, recalls[-1])
        save_recall(args.dataset, args.model, recalls, args.eval_term)