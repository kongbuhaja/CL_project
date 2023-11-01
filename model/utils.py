import torch, os

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

def save_model(model, dataset_name, model_name, epoch, loss, ap=0):
    path = dir_check(dataset_name)
    
    model_path = path + '/.pt'
    info_path = path + '/.info'
    torch.save(model, model_path)
    with open(info_path, 'w') as f:
        text = f'epoch:{epoch}\n' +\
               f'loss:{loss}\n' +\
               f'ap:{ap}'
        f.write(text)
    print(f'Success to save model in {model_path}')

def load_model(dataset_name, model_name, nclasses, nf=20, input_channels=3, load=False):
    model_path = f'./checkpoints/{dataset_name}/{model_name}/{model_name}.pt'
    info_path = f'./checkpoints/{dataset_name}/{model_name}/{model_name}.info'
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
    print(f'Create {model_name} model')
    model = Model(model_name, nclasses, nf=nf, input_channels=input_channels)
    return model, *[0, 9999999999, 0.]

# def load_models(dataset_name, model_names, nclasses, nf=20, input_channels=3, load=False):
#     model_names = [model_name for model_name in model_names.replace(' ' ,'').split(',')]
#     models, start_epochs, best_losses, aps = [], [], [], []
#     for model_name in model_names:
#         model, start_epoch, best_loss, ap = load_model(dataset_name, model_name, nclasses, nf, input_channels, load)
#         models += [model]
#         start_epochs += [start_epochs]
#         best_losses += [best_loss]
#         aps += [ap]
#     return model_names, models, start_epochs, best_losses, aps
