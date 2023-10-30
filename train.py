import torch
import cv2
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader

from model.utils import load_model, save_model
from data.utils import *
from utils import arg_parse

import numpy as np

np.random.seed(42)
torch.manual_seed(42)

args = arg_parse()
batch_size = args.batch_size
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Device = torch.device("cpu")

train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model, start_epochs, best_loss, ap = load_model(args.dataset, args.model, len(train_dataset.unique_labels), 100, load=args.load)

epochs = start_epochs + args.epochs
train_iters = train_dataset.length//batch_size
val_iters = val_dataset.length//batch_size

if args.loss == 'CE':
    loss_fn = torch.nn.CrossEntropyLoss()
elif args.loss == 'BCE':
    loss_fn = lambda x, y :torch.nn.BCEWithLogitsLoss()(x, 
                torch.nn.functional.one_hot(y, len(train_dataset.unique_labels)).type(torch.float32))

model.to(Device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

for epoch in range(start_epochs, epochs):
    # train
    train_loss = 0.
    for iter, (x_data, y_data) in enumerate(train_dataloader):
        pred = model(x_data.to(Device))
        loss = loss_fn(pred, y_data[..., 0].to(Device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print(f'epoch: {epoch+1}/{epochs}, iter:{iter+1}/{train_iters}| total_loss: {train_loss/(iter+1):.3f}', flush=True, end='\r',)
    print(f'epoch: {epoch+1}/{epochs}, iter:{iter+1}/{train_iters}| total_loss: {train_loss/train_iters:.3f}')

    # eval
    with torch.no_grad():
        if epoch%5==0 or start_epochs==epochs-1:
            val_loss = 0.
            for iter, (x_data, y_data) in enumerate(val_dataloader):
                pred = model(x_data.to(Device))
                loss = loss_fn(pred, y_data[..., 0].to(Device))

                val_loss += loss.item()
                # need to make evaluate metrics
                print(f'epoch: {epoch+1}/{epochs}, iter:{iter+1}/{val_iters}| val_loss: {val_loss/(iter+1):.3f}', flush=True, end='\r')
            print(f'epoch: {epoch+1}/{epochs}, iter: {iter+1}/{val_iters}| val_loss: {val_loss/val_iters:.3f}')

        if val_loss/train_iters < best_loss:
            best_loss = val_loss/train_iters
            save_model(model, args.dataset, args.model, epoch, best_loss, 0.)
            
# temporary code
test_x_data = []
test_y_data = []
for x_data, y_data in val_dataloader:
    for x, y in zip(x_data, y_data):
        test_x_data += [x]
        test_y_data += [y]
        if len(test_x_data) > 8:
            break
    if len(test_x_data) > 8:
        break

row = None
output = None

for i, (x, y) in enumerate(zip(test_x_data, test_y_data)):
    with torch.no_grad():
        pred = np.argmax(model(x[None].to(Device)).cpu().numpy(), -1)
    x = (x.numpy()*255).astype(np.uint8)
    y = y.numpy()
    if x.shape[-1] == 1:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    x = cv2.resize(x, [100, 100])
    x = cv2.putText(x, str(pred[0]), (0,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    x = cv2.putText(x, str(y[0]), (82,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    if row is None:
        row = x
    else:
        row = np.concatenate([row, x], 1)
        if i%3 == 2:
            if output is None:
                output = row
            else:
                output = np.concatenate([output, row], 0)
            row = None

cv2.imwrite('image.jpg', output)
# cv2.imshow('image', output)
# cv2.waitKey()
# cv2.destroyAllWindows()

# loss 계산시 nan처리