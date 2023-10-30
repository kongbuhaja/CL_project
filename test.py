import torch
import cv2

from model.utils import load_model, save_model
from data.utils import *
from utils import arg_parse

import torchsummary
import numpy as np

args = arg_parse()
# Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Device = torch.device("cpu")

# d_tr, n_inputs, n_outputs, len_dtr, input_channels = load_datasets(args.dataset)
# height = width = int(np.sqrt(n_inputs//input_channels))
# len_per_dtr = len(d_tr[0][1])

model = load_model(args.dataset, args.model, 10, 20).to(Device)

torchsummary.summary(model, (28,28,3), device='cpu')

# for data in d_tr:
#     y = data[2]
#     print(torch.bincount(y))

# for data in d_tr:
#     x = data[1]
#     y = data[2]
#     xx = x[0].view(32, 32, 3).numpy()
#     yy = y[0]
#     cv2.imshow(f'{yy}', xx)
#     cv2.waitKey()
# cv2.destroyAllWindows()

