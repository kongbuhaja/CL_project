import argparse
import torch

def arg_parse():
    parser = argparse.ArgumentParser(description='base model')

    parser.add_argument('--model', dest='model', type=str, default='DarkNet19', help='model to train')
    parser.add_argument('--loss' , dest='loss', type=str, default='CE', help='function to train model')
    parser.add_argument('--load', dest='load', type=bool, default=False, help='whether to load model')
    parser.add_argument('--epochs', dest='epochs', type=int, default=30, help='epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1000, help='batch_size for training or inference')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning rate for training')

    parser.add_argument('--dataset', dest='dataset', type=str, default='mnist', help='dataset for training')
    parser.add_argument('--image_size', dest='image_size', type=str, default='128x128', help='dataset for training')

    args = parser.parse_args()
    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'loss: {args.loss}')

    return args

class LR_schedular:
    def __init__(self, optimizer, schedule):
        self.optimizer = optimizer
        self.init_lr = self.optimizer.param_groups[0]['lr']
        self.lr = self.init_lr
        
        if schedule == 'static':
            self.schedule = self.static
        elif schedule == 'linear':
            self.schedule = self.linear_decay

        self.optimizer.zero_grad()

    def step(self, *params):
        self.lr = self.schedule(*params)
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.lr
        self.optimizer.step()
        self.optimizer.zero_grad()

    def linear_decay(self, iter, iters, eps=1e-9):
        return max(self.init_lr * (1 - iter / iters), eps)

    def static(self, *params):
        return self.init_lr

def loss_function(method, last_dims):
    if method.upper() == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif method.upper() == 'BCE':
        loss_fn = lambda x, y :torch.nn.BCEWithLogitsLoss()(x, 
                    torch.nn.functional.one_hot(y, last_dims).type(torch.float32))
    return loss_fn