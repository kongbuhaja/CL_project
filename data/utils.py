import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os, shutil, cv2
import numpy as np
from data.augmentation import *
from torchvision import transforms


# need to make transform
class Custom_Dataset(Dataset):
    def __init__(self, path, image_size, transfrom=True):
        self.image_size = image_size
        self.images = []
        self.labels = []
        for label in os.listdir(path):
            files = list(map(lambda file: f'{path}/{label}/{file}', os.listdir(f'{path}/{label}')))
            self.images += files
            self.labels += [[int(label)]] * len(files)

        self.unique_labels = np.unique(self.labels).tolist()
        self.length = len(self.images)

        if transfrom:
            self.transforms = transforms.Compose([Padding(), 
                                                  Resize(self.image_size),
                                                  Normalization()])
        else:
            self.transforms = Copy()

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = self.transforms(image)
        image = torch.FloatTensor(image)
        label = torch.LongTensor(self.labels[idx])

        return image, label
    
    def __len__(self):
        return len(self.images)
    
def load_dataset(data_name, image_size, path='data'):
    path = f'{path}/{data_name.upper()}'
    train_path = f'{path}/train'
    val_path = f'{path}/val'

    if not os.path.exists(path):
        os.makedirs(path)
        download_dataset(data_name, path=path)
    elif not os.path.exists(train_path) and not os.path.exists(val_path):
        download_dataset(data_name, path=path)
    
    train_dataset = Custom_Dataset(train_path, image_size)
    val_dataset = Custom_Dataset(val_path, image_size)
    
    return train_dataset, val_dataset

def download_dataset(data_name, path):
    if data_name=='mnist':
        from_datasets = datasets.MNIST
    elif data_name=='imagenet':
        from_datasets = datasets.ImageNet
        # to be continue
    elif data_name=='cifar100':
        from_datasets = datasets.CIFAR100

    train_dataset = from_datasets(root=path,
                                download=True,
                                transform = transforms.ToTensor())
    val_dataset = from_datasets(root=path,
                                train = False,
                                download=True,
                                transform = transforms.ToTensor())
    save_dataset(data_name, train_dataset, val_dataset, path=path)

def save_dataset(data_name, train_dataset, val_dataset, path='data'):
    if data_name in ['mnist', 'cifar100']:
        for dir in os.listdir(path):
            path_ = f'{path}/{dir}'
            if os.path.isdir(path_):
                shutil.rmtree(path_)
            else:
                os.remove(path_)
    
    labels = torch.unique(torch.Tensor(train_dataset.targets))
    for task in ['train', 'val']:
        for label in labels:
            os.makedirs(f'{path}/{task}/{int(label)}')

    for task in ['train', 'val']:
        indices = [0] * len(labels)
        dataset = train_dataset if task=='train' else val_dataset
        for img, label in zip(np.array(dataset.data), dataset.targets):
            cv2.imwrite(f'{path}/{task}/{int(label)}/{indices[label]}.jpg', img)
            indices[label] += 1