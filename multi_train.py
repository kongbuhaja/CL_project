# import torch
# import cv2
# from torchvision.transforms import Lambda
# from torch.utils.data import DataLoader

# from model.utils import load_model, save_model, load_models
# from data.utils import *
# from utils import arg_parse, loss_function, LR_schedular

# import numpy as np

# np.random.seed(42)
# torch.manual_seed(42)

# args = arg_parse()

# Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Device = torch.device("cpu")

# train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# model_names, models, start_epochs, best_losses, aps = load_models(args.dataset, args.model, len(train_dataset.unique_labels), 100, load=args.load)
# for i in range(len(models)):
#     models[i].to(Device)

# epochs = args.epochs
# train_iters = train_dataset.length//args.batch_size
# val_iters = val_dataset.length//args.batch_size

# optimizer = torch.optim.SGD([{'params': models[i].parameters()} for i in range(len(models))], lr=args.lr, momentum=0.9)
# scheduler = LR_schedular(optimizer, 'linear')
# loss_fn = loss_function(args.loss, len(train_dataset.unique_labels))

# for epoch in range(1, epochs+1):
#     # train
#     train_losses = [0.] * len(models)
#     lrs = []
#     for iter, (x_data, y_data) in enumerate(train_dataloader):
#         text = ''
#         for i in range(len(models)):
#             pred = models[i](x_data.to(Device))
#             loss = loss_fn(pred, y_data[..., 0].to(Device))
#             loss.backward()
#             scheduler.step((start_epochs[i]+epoch)*train_iters+iter+1, (start_epochs[i]+epochs)*train_iters, args.lr/10)
            
#             lrs[i] = scheduler.lr
#             train_losses[i] += loss.item()
#             text += f'model: {model_names[i]}, epoch: {start_epochs[i]+epoch}/{start_epochs+epochs}, iter: {iter+1}/{train_iters} | lr: {lrs[i]:.4f}, total_loss: {train_losses[i]/(iter+1):.4f}\n'
#         print(text[:-2], flush=True, end='\r',)
#     text = ''
#     for i in len(models):
#         text += f'model: {model_names[i]}, epoch: {start_epochs[i]+epoch}/{start_epochs[i]+epochs}, iter: {iter+1}/{train_iters} | lr: {lrs[i]:.4f}, total_loss: {train_losses[i]/train_iters:.4f}\n'
#     print(text[:-2])

#     # eval
#     with torch.no_grad():
#         if epoch%5==0 or start_epochs==epochs-1:
#             val_losses = [0.] * len(models)
#             for iter, (x_data, y_data) in enumerate(val_dataloader):
#                 text = ''
#                 for i in len(models):
#                     pred = models[i](x_data.to(Device))
#                     loss = loss_fn(pred, y_data[..., 0].to(Device))

#                     val_losses[i] += loss.item()
#                     # need to make evaluate metrics
#                     text += f'model: {model_names[i]}, epoch: {start_epochs[i]+epoch}/{start_epochs+epochs}, iter: {iter+1}/{val_iters} | val_loss: {val_losses[i]/(iter+1):.4f}\n'
#                 print(text[:-2], flush=True, end='\r')
#             text = ''
#             for i in len(models):
#                 text += f'model: {model_names[i]}, epoch: {start_epochs[i]+epoch}/{start_epochs[i]+epochs}, iter: {iter+1}/{val_iters} | val_loss: {val_losses[i]/val_iters:.4f}\n'
#             print(text[:-2])

#             for i in len(models):
#                 if val_losses[i]/val_iters < best_loss:
#                     best_loss = val_losses[i]/val_iters
#                     save_model(models, args.dataset, model_names[i], epoch, best_loss, 0.)
            