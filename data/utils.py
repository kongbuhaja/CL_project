import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os, shutil, cv2
import numpy as np
from data.augmentation import *
from torchvision import transforms
import sys, urllib.request, tarfile
from data.dataset import Origin_Dataset, Continuum_Dataset

def load_dataset(args):
    data_name = args.dataset.upper()
    train_path = f'{args.data_path}/{data_name}/train'
    val_path = f'{args.data_path}/{data_name}/val'

    if not os.path.exists(train_path) and not os.path.exists(val_path):
        make_dataset(data_name, args.data_path)

    if not args.continuum:
        # print(args.continuum)
        train_dataset = Origin_Dataset(train_path, args.image_size, args.in_channel)
        val_dataset = Origin_Dataset(val_path, args.image_size, args.in_channel, transform=False)
    else:
        # print('2')
        train_dataset = Continuum_Dataset(train_path, args.n_tasks, args.image_size, args.in_channel)
        val_dataset = Continuum_Dataset(val_path, args.n_tasks, args.image_size, args.in_channel, transform=False)

    return train_dataset, val_dataset

def download_from_server(data_name, path, ip_address='166.104.144.76', port=8000):
    url = f'http://{ip_address}:{port}/Classification/{data_name}_RAW.tar.gz'
    urllib.request.urlretrieve(url, f'{path}/{data_name}_RAW.tar.gz')

def extract(data_name, path):
    with tarfile.open(f'{path}/{data_name}_RAW.tar.gz', 'r:gz') as tar:
        tar.extractall(f'data/')

def make_dataset(data_name, path):
    data_path = f'{path}/{data_name}'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if data_name=='MNIST':
        from_datasets = datasets.MNIST
    elif data_name=='IMAGENET':
        from_datasets = datasets.ImageNet
    elif data_name=='CIFAR100':
        from_datasets = datasets.CIFAR100
    elif data_name=='CIFAR10':
        from_datasets = datasets.CIFAR10

    if data_name=='IMAGENET':
        raw_data_path = f'{data_path}_RAW'
        if not os.path.exists(raw_data_path):
            try:
                print(f'{raw_data_path} is not exist')
                download_from_server(data_name, path)
                extract(data_name, path)
            except:
                print(f'You need to download imagenet dataset and rename {data_name}_RAW')
                sys.exit(0)
        else:
            print(f'{raw_data_path} is exist')
        train_dataset = from_datasets(root=raw_data_path,
                                      split='train')
        val_dataset = from_datasets(root=raw_data_path,
                                    split='val')
        save_imagenet(train_dataset, val_dataset, data_path)
    else:
        if data_name in ['MNIST']:
            train_dataset = from_datasets(root=data_path,
                                        download=True)
            val_dataset = from_datasets(root=data_path,
                                        train = False,
                                        download=True)

        elif data_name in ['CIFAR100', 'CIFAR10']:
            train_dataset = from_datasets(root=data_path,
                                        download=True)

            val_dataset = from_datasets(root=data_path,
                                        train = False,
                                        download=True)
        save_dataset(data_name, train_dataset, val_dataset, data_path)



def save_dataset(data_name, train_dataset, val_dataset, path):
    if data_name in ['MNIST', 'CIFAR100']:
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
            img = img[..., [2, 1, 0]] if len(img.shape)==3 else img
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

    # for dir in os.listdir(f'{path}_RAW'):
    #     if os.path.isdir(f'{path}_RAW/{dir}'):
    #         shutil.rmtree(f'{path}_RAW/{dir}')