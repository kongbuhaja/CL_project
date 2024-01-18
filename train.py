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

model, start_epoch, best_recall, recalls, save_path = load_model(args.dataset, args.optimizer, args.model, args.channel, 
                                                                 len(train_dataset.unique_labels), args.eval_term, load=args.load)
model.to(Device)

epochs = args.epochs
train_iters = len(train_dataloader)

optimizer = get_optimizers(model, args.optimizer, args.init_lr)

scheduler = LR_schedular(optimizer, args.lr_schedular)
loss_fn = loss_function(args.loss, len(train_dataset.unique_labels))

losses = [[], []]
for epoch in range(start_epoch, epochs):
    # train
    train_loss = 0.
    model.train()
    train_tqdm = tqdm.tqdm(train_dataloader, total=train_iters, ncols=121, desc=f'Train epoch {epoch+1}/{epochs}', ascii=' =', colour='red')
    for iter, (x_data, y_data) in enumerate(train_tqdm):
        pred = model(x_data.to(Device))
        loss = loss_fn(pred, y_data[..., 0].to(Device))
        loss.backward()
        scheduler.step(epoch*train_iters+iter, epochs*train_iters)

        train_loss += loss.item()
        train_tqdm.set_postfix_str(f'| lr: {scheduler.lr:.4f}, total_loss: {train_loss/(iter+1):.4f}')
    losses[0] += [train_loss/(iter+1)]
    
    # eval
    if (epoch+1)%args.eval_term==0:
        recall, loss = eval(model, val_dataloader, loss_fn, Device)
        recalls += [recall]
        losses[1] += [loss]
        if best_recall < recalls[-1]:
            best_recall = recalls[-1]
            save_model(model, save_path, epoch+1, recalls[-1])
        save_recall(save_path, recalls, args.eval_term)
    save_loss(save_path, losses, args.eval_term)