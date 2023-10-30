import torch

from model.Darknet19_origin import DarkNet19
from model.Resnet18_modified import ResNet18
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

def save_model(model, dataset_name, model_name, epochs, loss, ap=0):
    dir_check(dataset_name)
    model_path = f'./checkpoints/{dataset_name}/{model_name}.pt'
    info_path = f'./checkpoints/{dataset_name}/{model_name}.info'
    torch.save(model, model_path)
    with open(info_path, 'w') as f:
        text = f'epochs:{epochs}\n\
                 loss:{loss}\n\
                 ap:{ap}'
        f.write(text)
    print(f'Success to save model in {model_path}')

def load_model(dataset_name, model_name, nclasses, nf=20, input_channels=3, load=False):
    model_path = f'./checkpoints/{dataset_name}/{model_name}.pt'
    info_path = f'./checkpoints/{dataset_name}/{model_name}.info'
    try:
        if load:
            model = torch.load(model_path)
            with open(info_path, 'r') as f:
                text = f.readlines()
            result = [info.split('\n')[0].split(':')[1] for info in text]
            result = [int(re) if '.' not in re else float(re) for re in result]
            print(f'Success to load model from {model_path}')
            return model, *result
    except:
        print(f'Fail to load model from {model_path}, So ', end='')
    print('Create new model')
    model = Model(model_name, nclasses, nf=nf, input_channels=input_channels)
    return model, *[0, 9999999999, 0.]