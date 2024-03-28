import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os, cv2
from data.augmentation import *

class Base_Dataset(Dataset):
    def __init__(self, path, image_size, in_channel, transform=True):
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(path))
        self.unique_labels = [l for l in range(len(self.classes))]
        self.color = cv2.IMREAD_COLOR if in_channel==3 else cv2.IMREAD_GRAYSCALE
        self.color_func = (lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) if in_channel==3 else (lambda x: x[..., None])

        if transform:
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
        image = self.color_func(cv2.imread(self.images[idx], self.color))
        image = self.transforms(image)
        image = torch.FloatTensor(image)
        label = torch.LongTensor(self.labels[idx])

        return image, label

    def __len__(self):
        return len(self.images)

class Origin_Dataset(Base_Dataset):
    def __init__(self, path, image_size, in_channel, transform=True):
        super().__init__(path, image_size, in_channel, transform)
        for label, c in enumerate(self.classes):
            files = list(map(lambda file: f'{path}/{c}/{file}', os.listdir(f'{path}/{c}')))
            self.images += files
            self.labels += [[int(label)]] * len(files)

class Continuum_Dataset(Base_Dataset):
    def __init__(self, path, n_tasks, image_size, in_channel, transform=True):
        super().__init__(path, image_size, in_channel, transform)
        self.task = 0
        self.n_tasks = n_tasks

        for task in range(n_tasks):
            class_per_task = int(np.ceil(len(self.classes) / n_tasks))
            start = class_per_task * task
            end = class_per_task * (task + 1)
            task_images, task_labels = [], []
            for label, c in enumerate(self.classes[start:end]):
                files = list(map(lambda file: f'{path}/{c}/{file}', os.listdir(f'{path}/{c}')))
                task_images += files
                task_labels += [[start + int(label)]] * len(files)
            self.images += [task_images]
            self.labels += [task_labels]

    def __getitem__(self, idx):
        image = self.color_func(cv2.imread(self.images[self.task][idx], self.color))
        image = self.transforms(image)
        image = torch.FloatTensor(image)
        label = torch.LongTensor(self.labels[self.task][idx])

        return image, label

    def __len__(self):
        return len(self.images[self.task])