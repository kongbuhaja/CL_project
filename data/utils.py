import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os, shutil, cv2
import numpy as np
from data.augmentation import *
from torchvision import transforms
import sys


# need to make transform
class Custom_Dataset(Dataset):
    def __init__(self, path, image_size, transfrom=True):
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(path))
        self.unique_labels = [l for l in range(len(self.classes))]

        for label, c in enumerate(self.classes):
            files = list(map(lambda file: f'{path}/{c}/{file}', os.listdir(f'{path}/{c}')))
            self.images += files
            self.labels += [[int(label)]] * len(files)

        self.length = len(self.images)

        if transfrom:
            self.transforms = transforms.Compose([Random_resize(self.image_size),
                                                  Padding(),
                                                  Rotate90(),
                                                  Random_Vflip(),
                                                  Normalization()])
        else:
            self.transforms = transforms.Compose([Resize(self.image_size),
                                                  Padding(),
                                                  Normalization()])

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
    val_dataset = Custom_Dataset(val_path, image_size, transfrom=False)
    
    return train_dataset, val_dataset

def download_dataset(data_name, path):
    if data_name=='mnist':
        from_datasets = datasets.MNIST
    elif data_name=='imagenet':
        from_datasets = datasets.ImageNet
    elif data_name=='cifar100':
        from_datasets = datasets.CIFAR100

    if data_name in ['mnist', 'cifar100']:
        train_dataset = from_datasets(root=path,
                                    download=True,
                                    transform = transforms.ToTensor())
        val_dataset = from_datasets(root=path,
                                    train = False,
                                    download=True,
                                    transform = transforms.ToTensor())
        save_dataset(data_name, train_dataset, val_dataset, path)
    elif data_name=='imagenet':
        if not os.path.exists(f'{path}_RAW'):
            os.makedirs(f'{path}_RAW')
            print('You need download imagenet dataset directly in {path}_RAW')
            sys.exit(0)
        train_dataset = from_datasets(root=f'{path}_RAW',
                                      split='train',
                                      transform = transforms.ToTensor())
        val_dataset = from_datasets(root=f'{path}_RAW',
                                    split='val',
                                    transform = transforms.ToTensor())
        save_imagenet(train_dataset, val_dataset, path)
    

def save_dataset(data_name, train_dataset, val_dataset, path):
    if data_name in ['mnist', 'cifar100']:
        for dir in os.listdir(path):
            path_ = f'{path}/{dir}'
            if os.path.isdir(path_):
                shutil.rmtree(path_)
            else:
                os.remove(path_)
    
    classes = train_dataset.classes
    for task in ['train', 'val']:
        for c in classes:
            os.makedirs(f'{path}/{task}/{c}')

    for task, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
        indices = [0] * len(classes)
        for img, label in zip(np.array(dataset.data), dataset.targets):
            cv2.imwrite(f'{path}/{task}/{classes[label]}/{indices[label]}.jpg', img)
            indices[label] += 1
        print(f'{task} data| total:{np.sum(indices)}, mean:{np.mean(indices)}, std:{np.std(indices)}')

def save_imagenet(train_dataset, val_dataset, path):
    classes = []
    for c in train_dataset.classes:
        if c[0] in classes:
            count = classes.count(c[0])
            classes += [f'{c[0]}{count+1}']
        else:
            classes += [c[0]]

    for task in ['train', 'val']:
        for c in classes:
            os.makedirs(f'{path}/{task}/{c}')

    for task, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
        indices = [0] * len(classes)
        for file, label in dataset.imgs:
            moved_file = f'{path}/{task}/{classes[label]}/{indices[label]}.jpg'
            shutil.move(file, moved_file)
            indices[label] += 1
        print(f'{task} data| total:{np.sum(indices)}, mean:{np.mean(indices)}, std:{np.std(indices)}')

    for dir in os.listdir(f'{path}_RAW'):
        if os.path.isdir(f'{path}_RAW/{dir}'):
            shutil.rmtree(f'{path}_RAW/{dir}')
    
