import torch, tqdm
from torch.utils.data import DataLoader

from model.utils import *
from data.utils import *
from utils import *

args = args_parse()
args_show(args)
env_set(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset = load_dataset(args, continuum=False)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.cpus)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.cpus)

model, start_epoch, best_recall, recalls, save_path = load_model(args, len(train_dataset.unique_labels), device=device)

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
        scheduler.step(epoch*train_iters+iter, epochs*train_iters)
        optimizer.zero_grad()

        pred = model(x_data.to(device).permute(0,3,1,2))
        loss = loss_fn(pred, y_data[..., 0].to(device))
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_tqdm.set_postfix_str(f'| lr: {scheduler.lr:.4f}, total_loss: {train_loss/(iter+1):.4f}')
    losses[0] += [train_loss/(iter+1)]
    
    # eval
    if (epoch+1)%args.eval_term==0:
        model.eval()
        with torch.no_grad():
            positive = 0
            val_loss = 0.
            val_tqdm = tqdm.tqdm(val_dataloader, total=len(val_dataloader), ncols=121, desc=f'Validation', ascii=' =', colour='blue')
            for iter, (x_data, y_data) in enumerate(val_tqdm):
                pred = model(x_data.to(device).permute(0,3,1,2))
                loss = loss_fn(pred, y_data[..., 0].to(device))

                val_loss += loss.item()
                
                pred_label = torch.argmax(pred, -1).to('cpu')
                positive += sum(pred_label == y_data[..., 0])
                recall = positive/((iter+1)*val_dataloader.batch_size)
                val_tqdm.set_postfix_str(f'| recall: {recall:.3f}, val_loss: {val_loss/(iter+1):.4f}')

        recalls += [recall.numpy()]
        losses[1] += [val_loss/(iter+1)]
        if best_recall < recalls[-1]:
            best_recall = recalls[-1]
            save_model(model, save_path, epoch+1, recalls[-1])
        save_recall(save_path, recalls, args.eval_term)
    save_loss(save_path, losses, args.eval_term)