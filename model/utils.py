import torch, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from model.DarkNet import DarkNet19
from model.ResNet import ResNet18
from model.VGG import VGG19, VGG16
from model.GoogleNet import GoogleNet22
from model.MLP import MLP
from utils import dir_check

def Model(model_name, channel, n_classes, image_size, in_channel):
    if model_name == 'DarkNet19':
        model = DarkNet19(channel, n_classes, in_channel)
    elif model_name == 'ResNet18':
        model = ResNet18(channel, n_classes, in_channel)
    elif model_name == 'VGG16':
        model = VGG16(channel, n_classes, image_size, in_channel)
    elif model_name == 'VGG19':
        model = VGG19(channel, n_classes, image_size, in_channel)
    elif model_name == 'GoogleNet22':
        model = GoogleNet22(channel, n_classes, image_size, in_channel)
    elif model_name == 'MLP':
        model = MLP(channel, n_classes, in_channel)

    return model

def save_model(model, path, epoch, recall):  
    model_path = f'{path}/model.pt'
    info_path = f'{path}/model.info'
    torch.save(model, model_path)
    with open(info_path, 'w') as f:
        text = f'epoch:{epoch}\n' +\
               f'recall:{recall}\n'
        f.write(text)
    print(f'Success to save model in {model_path}')

def load_model(dataset_name, optimizer, model_name, channel, nclasses, image_size, in_channel=3, load=False, checkpoint='checkpoints'):
    dir = dir_check(f'{checkpoint}/{dataset_name}/{optimizer}/{model_name}')
    dir_check(f'{dir}/result')

    model_path = dir + '/model.pt'
    info_path = dir + '/model.info'
    recall_path = dir + '/result/recall.txt'

    if load:
        try:
            model = torch.load(model_path)
            with open(info_path, 'r') as f:
                text = f.readlines()
            result = [info.split('\n')[0].split(':')[1] for info in text]
            result = [int(re) if '.' not in re else float(re) for re in result]
            
            with open(recall_path, 'r') as f:
                text = f.readline()
            recalls = [float(r) for r in text.split(' ')[:-1]]

            print(f'Success to load model from {model_path}')
            return model, *result, recalls, dir
        except:
            print(f'Fail to load model from {model_path}, So ', end='')

    print(f'Create {model_name} model')
    model = Model(model_name, channel, nclasses, image_size, in_channel)
    return model, *[0, 0.], [], dir

def save_recall(path, recalls, term):
    path += '/result'

    with open(f'{path}/recall.txt', 'w') as f:
        for r in recalls:
            f.write(f'{r} ')
    
    dirs = path.split('/')
    save_graph('recall', path, recalls, term, f'{dirs[-4]} X {dirs[-3]} X {dirs[-2]}', 'top')

def save_loss(path, losses, term):
    path += '/result'

    dirs = path.split('/')
    save_graph('loss_train', path, losses[0], 1, f'{dirs[-4]} X {dirs[-3]} X {dirs[-2]}', 'bottom')
    if len(losses[0])/term == len(losses[1]):
        save_graph('loss_eval', path, losses[1], term, f'{dirs[-4]} X {dirs[-3]} X {dirs[-2]}', 'bottom')

def save_graph(task, dir_path, values, term, title, highlight='top'):
    epochs = [e for e in range(1, len(values)+1)]
    stamp = max(epochs[-1]//10, 1)
    xticks = [str(e*term)if e%stamp==0 else '' for e in epochs]

    plt.clf()
    plt.xlabel('epoch')
    plt.xticks(epochs,xticks)
    plt.ylabel(task)
    plt.title(title)

    width = epochs[-1]
    height = max(values) - min(values)

    max_idx = np.argmax(values, -1)
    min_idx = np.argmin(values, -1)
    plt.text(epochs[max_idx]-width/16, values[max_idx]+height/len(values)*2*0.01, f'{values[max_idx]:.5f}')
    plt.text(epochs[min_idx]-width/16, values[min_idx]+height/len(values)*2*0.01, f'{values[min_idx]:.5f}')

    plt.plot(epochs, values)
    if highlight:
        idx = max_idx if highlight=='top' else min_idx
        circle = patches.Ellipse((epochs[idx]-width/16*0.22, values[idx] + height*2*0.01), 
                                width = width/6,
                                height = height/7*0.5,
                                edgecolor = 'red', 
                                linestyle = 'dotted',
                                linewidth = 2,
                                fill = False)
        plt.gca().add_patch(circle)
    
    plt.savefig(f'{dir_path}/{task}.jpg')