import torch
import cv2
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader

from model.utils import load_model, save_model
from data.utils import *
from utils import arg_parse

import numpy as np

args = arg_parse()
batch_size = args.batch_size
# Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Device = torch.device("cpu")

_, val_dataset = load_dataset(args.dataset)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = load_model(args.dataset, args.model, len(val_dataset.unique_labels), 20, load=True).to(Device)

val_iters = val_dataset.length//batch_size

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
    pred = np.argmax(model(x[None]/255.).detach().cpu().numpy(), -1)
    x = x.cpu().numpy().astype(np.uint8)
    y = y.cpu().numpy()
    if x.shape[-1] == 1:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    x = cv2.resize(x, [112, 112])
    x = cv2.putText(x, str(pred[0]), (0,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    x = cv2.putText(x, str(y[0]), (92,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
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

show = True
if True:
    cv2.imshow('image', output)
    cv2.waitKey()
    cv2.destroyAllWindows()
cv2.imwrite('image.jpg', output)

print(torch.max(test_x_data[0]))
