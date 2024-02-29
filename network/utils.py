import torch, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from network.GEM import GEM
from network.Single import Single
from utils import dir_check

def Network(args, n_classes, device):
    if args.network in ['single', 'Single']:
        network = Single(args, n_classes, device=device)
    elif args.network in ['gem', 'GEM']:
        network = GEM(args, n_classes, device=device)
    return network

def save_model(model, path, epoch, recall):  
    model_path = f'{path}/model.pt'
    info_path = f'{path}/model.info'
    torch.save(model, model_path)
    with open(info_path, 'w') as f:
        text = f'epoch:{epoch}\n' +\
               f'recall:{recall}\n'
        f.write(text)
    print(f'Success to save model in {model_path}')

def load_network(args, n_classes, device='cpu'):
    dir = dir_check(f'{args.checkpoint}/{args.dataset}/{args.optimizer}/{args.network}/{args.model}')

    network = Network(args, n_classes, device)
    return network, 0, 0., [], dir

def save_stats(path, values, recalls, labels):
    path += '/result'
    dir_check(path)
    
    dirs = path.split('/')
    title = f'{dirs[-5]} X {dirs[-3]} X {dirs[-2]}'
    save_graph(path, values, labels, title)
    save_heat_map(path, recalls, title)

def save_graph(dir_path, values, labels, title):
    tasks = [task for task in range(1, len(values[0])+1)]
    colors = [np.random.uniform(0, 0.7, 3) for _ in range(len(labels))]

    plt.xlabel('tasks')
    plt.xticks(tasks)
    plt.ylabel('acc')
    plt.yticks(np.arange(-0.5, 1.1, 0.1))
    plt.ylim([-0.5, 1.0])
    plt.title(title)

    for value, label, color in zip(values, labels, colors):
        plt.plot(tasks, value, color=color, label=label)
    
    for value, color in zip(values, colors):
        plt.text(tasks[-1]-1+0.85, value[-1]+0.01, f'{value[-1]:.3f}', color=color)
    
    plt.legend()

    plt.savefig(f'{dir_path}/graph.jpg')
    plt.close()

def save_heat_map(dir_path, values, title):
    cmap = plt.get_cmap('Reds')

    plt.matshow(values, cmap=cmap, aspect='auto')
    plt.title(title+'(ACC)')
    plt.xticks(np.arange(0, len(values[0])))
    plt.xlabel('tasks')
    plt.yticks(np.arange(0, len(values)))
    plt.ylabel('tasks')

    for (i, j), val in np.ndenumerate(values):
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=8)

    plt.colorbar()
    plt.clim(0.0, 1.0)
    plt.savefig(f'{dir_path}/heat_map.jpg')
    plt.close()
