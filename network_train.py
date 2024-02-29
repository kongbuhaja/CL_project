import torch, tqdm
from torch.utils.data import DataLoader

from network.utils import *
from data.utils import *
from utils import *

args = args_parse()
args_show(args)
env_set(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset = load_dataset(args)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.cpus)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=args.cpus)

network, start_epoch, best_recall, recalls, save_path = load_network(args, len(train_dataset.unique_labels), device=device)

epochs = args.epochs

optimizer = get_optimizers(network, args.optimizer, args.init_lr)

scheduler = LR_scheduler(optimizer, args.lr_scheduler)
loss_fn = loss_function(args.loss, len(train_dataset.unique_labels))

def train_each_task(task):
    network.train()
    train_dataset.task = task
    train_iters = len(train_dataloader)
    scheduler.init()
    for epoch in range(start_epoch, epochs):
        train_loss = 0.
        train_tqdm = tqdm.tqdm(train_dataloader, total=train_iters, ncols=121, desc=f'Train task {task+1}/{args.n_tasks} epoch {epoch+1}/{epochs}', ascii=' =', colour='red')
        for iter, (x_data, y_data) in enumerate(train_tqdm):
            scheduler.step(epoch*train_iters+iter, epochs*train_iters)
            optimizer.zero_grad()
            loss = network.observe(x_data.to(device).permute(0,3,1,2), task, y_data[..., 0].to(device))
            train_loss += loss.item()
            optimizer.step()
            train_tqdm.set_postfix_str(f'| lr: {scheduler.lr:.4f}, total_loss: {train_loss/(iter+1):.4f}')

def eval_all_task():
    network.eval()
    recall_vector = []
    with torch.no_grad():
        for val_task in range(args.n_tasks):
            val_dataset.task = val_task
            val_iters = len(val_dataloader)
            val_tqdm = tqdm.tqdm(val_dataloader, total=val_iters, ncols=121, desc=f'Validation task {val_task+1}/{args.n_tasks}', ascii=' =', colour='blue')
            positive = 0
            val_loss = 0.
            for iter, (x_data, y_data) in enumerate(val_tqdm):
                pred = network.model(x_data.to(device).permute(0,3,1,2))
                loss = loss_fn(pred, y_data[..., 0].to(device))

                val_loss += loss.item()
                
                pred_label = torch.argmax(pred, -1).to('cpu')
                positive += sum(pred_label == y_data[..., 0])
                recall = float(positive/((iter+1)*val_dataloader.batch_size))
                val_tqdm.set_postfix_str(f'| acc: {recall:.3f}, val_loss: {val_loss/(iter+1):.4f}')
            recall_vector += [recall]
    return recall_vector

stats = np.zeros([3, 0])
base_vector = np.array(eval_all_task())
for train_task in range(args.n_tasks):
    train_each_task(train_task)
    recalls += [eval_all_task()]
    
    recalls_np = np.array(recalls).reshape((-1, args.n_tasks))
    acc = np.mean(recalls_np[-1])
    bwt = np.mean(recalls_np[-1][:train_task] - np.diag(recalls_np)[:train_task]) if train_task > 0 else 0.
    fwt = np.mean(np.diag(recalls_np[..., 1:]) - base_vector[1:train_task+2])
    stats = np.concatenate([stats, np.stack([acc, bwt, fwt])[:, None]], -1)
    
    print(f'ACC: {acc:.3f}, BWT: {bwt:.3f}, FWT: {fwt:.3f}')
    save_stats(save_path, stats, recalls, ['ACC', 'BWT', 'FWT'])
    save_model(network.model, save_path, epochs*args.n_tasks, acc)
    print()