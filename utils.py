import argparse
import torch

def arg_parse():
    parser = argparse.ArgumentParser(description='base model')
    #model
    parser.add_argument('--model', dest='model', type=str, default='DarkNet19', help='model to train')
    parser.add_argument('--channel', dest='channel', type=int, default=16, help='channel of basis layers')
    parser.add_argument('--load', dest='load', type=str, default=False, help='whether to load model')
    
    #train
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=400, help='batch_size for training or inference')
    parser.add_argument('--loss' , dest='loss', type=str, default='CE', help='function to train model')
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=1e-2, help='init learning rate for training')
    parser.add_argument('--lr_schedular', dest='lr_schedular', type=str, default='linear', help='learning rate scheduler')

    #eval
    parser.add_argument('--eval_term', dest='eval_term', type=int, default=5, help='term of evaluate model')

    #dataset
    parser.add_argument('--dataset', dest='dataset', type=str, default='mnist', help='dataset for training')
    parser.add_argument('--image_size', dest='image_size', type=str, default='256x256', help='dataset for training')

    args = parser.parse_args()
    
    args.image_size = [int(l) for l in args.image_size.split('x')]
    args.load = args.load in ['True', 'true', 'T', 't']

    return args

def arg_print(args):
    print(f'dataset: {args.dataset}')
    print(f'image_size: {args.image_size[0]}x{args.image_size[1]}')
    print(f'batch_size: {args.batch_size}')
    print(f'model: {args.model}')
    print(f'channel: {args.channel}')
    print(f'loss: {args.loss}')

class LR_schedular:
    def __init__(self, optimizer, schedule):
        self.optimizer = optimizer
        self.init_lr = self.optimizer.param_groups[0]['lr']
        self.lr = self.init_lr
        self.min_lr = self.init_lr/100
        
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

    def linear_decay(self, iter, iters):
        return max(self.init_lr * (1 - iter / iters), self.min_lr)

    def static(self, *params):
        return self.lr

def loss_function(method, last_dims):
    if method.upper() == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif method.upper() == 'BCE':
        loss_fn = lambda x, y :torch.nn.BCEWithLogitsLoss()(x, 
                    torch.nn.functional.one_hot(y, last_dims).type(torch.float32))
    return loss_fn
