import torch, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from model.Darknet19 import DarkNet19
from model.Resnet18 import ResNet18
from model.MLP import MLP
from checkpoints.utils import dir_check

def Model(model_name, nclasses, nf=20, input_channels=3):
    if model_name == 'DarkNet19':
        model = DarkNet19(nclasses, nf, input_channels)
    elif model_name == 'ResNet18':
        model = ResNet18(nclasses, nf, input_channels)
    elif model_name == 'MLP':
        model = MLP(nclasses, nf, input_channels)

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

def load_model(dataset_name, model_name, nclasses, nf=20, input_channels=3, load=False):
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
    model = Model(model_name, nclasses, nf=nf, input_channels=input_channels)
    return model, *[0, 0.], []

def save_recall(dataset_name, model_name, recalls):
    epochs = [e*5 for e in range(1, len(recalls)+1)]
    dir_path = f'checkpoints/{dataset_name}/{model_name}' 
    with open(f'{dir_path}/recall.txt', 'w') as f:
        for r in recalls:
            f.write(f'{r} ')
    
    plt.xlabel('epochs')
    plt.xticks(epochs)
    plt.ylabel('recall')
    # plt.yticks([i*0.1 for i in range(10)])
    plt.title(f'{dataset_name} X {model_name} best recall : {max(recalls):.7f}')
    
    h_diff = (max(recalls)-min(recalls))*0.01
    w_diff = (max(epochs)-min(epochs))
    for e, r, in zip(epochs, recalls):
        plt.text(e-w_diff/len(epochs)/2, r+h_diff/len(recalls)*2, f'{r:.5f}')
    max_idx = np.argmax(recalls, -1)

    plt.plot(epochs, recalls)

    highlight = patches.Ellipse((epochs[max_idx], recalls[max_idx] + h_diff*2), 
                                 width = (max(epochs)-min(epochs))/len(epochs)*1.5,
                                 height = (max(recalls)-min(recalls))/len(recalls)/2,
                                 edgecolor = 'red', 
                                 linestyle = 'dotted',
                                 linewidth = 2,
                                 fill = False)
    plt.gca().add_patch(highlight)
    plt.savefig(f'{dir_path}/recall.jpg')
