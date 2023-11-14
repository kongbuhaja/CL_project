import torch, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from model.DarkNet import DarkNet19
from model.ResNet import ResNet18
from model.VGG import VGG19
from model.GoogleNet import GoogleNet22
from model.MLP import MLP
from checkpoints.utils import dir_check

def Model(model_name, channel, n_classes, image_size, in_channel):
    if model_name == 'DarkNet19':
        model = DarkNet19(channel, n_classes, in_channel)
    elif model_name == 'ResNet18':
        model = ResNet18(channel, n_classes, in_channel)
    elif model_name == 'VGG19':
        model = VGG19(channel, n_classes, image_size, in_channel)
    elif model_name == 'GoogleNet22':
        model = GoogleNet22(channel, n_classes, image_size, in_channel)
    elif model_name == 'MLP':
        model = MLP(channel, n_classes, in_channel)

    return model

def save_model(model, dataset_name, model_name, epoch, recall):
    path = dir_check(dataset_name, model_name)
    
    model_path = f'{path}/model.pt'
    info_path = f'{path}/model.info'
    torch.save(model, model_path)
    with open(info_path, 'w') as f:
        text = f'epoch:{epoch}\n' +\
               f'recall:{recall}\n'
        f.write(text)
    print(f'Success to save model in {model_path}')

def load_model(dataset_name, model_name, channel, nclasses, image_size, in_channel=3, load=False):
    dir = f'./checkpoints/{dataset_name}/{model_name}/'
    model_path = dir + 'model.pt'
    info_path = dir + 'model.info'
    recall_path = dir + 'recall.txt'
    try:
        if load:
            model = torch.load(model_path)
            with open(info_path, 'r') as f:
                text = f.readlines()
            result = [info.split('\n')[0].split(':')[1] for info in text]
            result = [int(re) if '.' not in re else float(re) for re in result]
            
            with open(recall_path, 'r') as f:
                text = f.readline()
            recalls = [float(r) for r in text.split(' ')[:-1]]

            print(f'Success to load model from {model_path}')
            return model, *result, recalls
    except:
        print(f'Fail to load model from {model_path}, So ', end='')

    print(f'Create {model_name} model')
    model = Model(model_name, channel, nclasses, image_size, in_channel)
    return model, *[0, 0.], []

def save_recall(dataset_name, model_name, recalls, eval_term, save_dir='checkpoints'):
    epochs = [e*eval_term for e in range(1, len(recalls)+1)]
    dir_path = f'{save_dir}/{dataset_name}/{model_name}' 

    if save_dir=='checkpoints':
        with open(f'{dir_path}/recall.txt', 'w') as f:
            for r in recalls:
                f.write(f'{r} ')
    plt.clf()
    plt.xlabel('epochs')
    plt.xticks(epochs)
    plt.ylabel('recall')
    plt.title(f'{dataset_name} X {model_name}')
    
    width = epochs[-1]
    height = max(recalls)-min(recalls)
    
    max_idx = np.argmax(recalls, -1)
    min_idx = np.argmin(recalls, -1)
    plt.text(epochs[max_idx]-width/16, recalls[max_idx]+height/len(recalls)*2*0.01, f'{recalls[max_idx]:.5f}')
    plt.text(epochs[min_idx]-width/16, recalls[min_idx]+height/len(recalls)*2*0.01, f'{recalls[min_idx]:.5f}')

    plt.plot(epochs, recalls)
    highlight = patches.Ellipse((epochs[max_idx]-width/16*0.22, recalls[max_idx] + height*2*0.01), 
                                 width = width/6,
                                 height = height/7*0.5,
                                 edgecolor = 'red', 
                                 linestyle = 'dotted',
                                 linewidth = 2,
                                 fill = False)
    plt.gca().add_patch(highlight)
    plt.savefig(f'{dir_path}/recall.jpg')
