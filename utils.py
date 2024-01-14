import argparse, torch, os, random
import numpy as np

def args_parse():
    parser = argparse.ArgumentParser(description='base model')
    #environment
    parser.add_argument('--gpus', dest='gpus', type=str, default='0', help='which device do you want to use')
    parser.add_argument('--cpus', dest='cpus', type=int, default=4, help='num of cpus for dataloader')

    #model
    parser.add_argument('--model', dest='model', type=str, default='DarkNet19', help='model to train')
    parser.add_argument('--channel', dest='channel', type=int, default=64, help='channel of basis layers')
    parser.add_argument('--load', dest='load', type=str, default=False, help='whether to load model')
    
    #train
    parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='epochs for training')
    parser.add_argument('--loss' , dest='loss', type=str, default='CE', help='function to train model')
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=1e-2, help='init learning rate for training')
    parser.add_argument('--lr_schedular', dest='lr_schedular', type=str, default='cosine_annealing', help='learning rate scheduler')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='Adam', help='optimizer for training')
    
    #eval
    parser.add_argument('--eval_term', dest='eval_term', type=int, default=5, help='term of evaluate model')

    #dataset
    parser.add_argument('--dataset', dest='dataset', type=str, default='cifar100', help='dataset for training')
    parser.add_argument('--image_size', dest='image_size', type=str, default='256x256', help='dataset for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='batch_size for training or inference')

    args = parser.parse_args()
    
    args.image_size = [int(l) for l in args.image_size.split('x')]
    args.load = args.load in ['True', 'true', 'T', 't']
    
    return args

def args_show(args, length=121, train=True):
    l = (length - 10) // 3

    print(f'=' * length)
    
    print(f'‖{"Environment":-^{length-2}}‖')
    print(f'‖ {"gpus: "+str(len(args.gpus.split(","))):<{l}} | {"cpus: "+str(args.cpus):<{l}} | {" ":<{l}} ‖')

    print(f'‖{"Train" if train else "Val":-^{length-2}}‖')
    print(f'‖ {"epochs: "+str(args.epochs):<{l}} | {"lr_schedular: "+args.lr_schedular:<{l}} | {"optimizer: "+args.optimizer+f"({str(args.init_lr)})":<{l}} ‖') if train else print('',end='')
    print(f'‖ {"loss: "+args.loss:<{l}} | {" ":<{l}} | {" ":<{l}} ‖')

    print(f'‖{"Model":-^{length-2}}‖')
    print(f'‖ {"model: "+args.model:<{l}} | {"channel: "+str(args.channel):<{l}} | {"load: "+str(args.load):<{l}} ‖')

    print(f'‖{"Data":-^{length-2}}‖')
    print(f'‖ {"dataset: "+args.dataset:<{l}} | {"image_size: "+str(args.image_size):<{l}} | {"batch_size: "+str(args.batch_size):<{l}} ‖')

    print(f'=' * length)

def env_set(gpus):
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

class LR_schedular:
    def __init__(self, optimizer, schedule):
        self.optimizer = optimizer
        self.init_lr = self.optimizer.param_groups[0]['lr']
        self.lr = self.init_lr
        self.min_lr = 1e-4
        
        if schedule == 'static':
            self.schedule = self.static
        elif schedule == 'linear':
            self.schedule = self.linear_decay
        elif schedule in 'cosine_annealing':
            self.schedule = self.cosine_annealing
            self.t_iter = 10
            self.t_max = self.t_iter
            self.t_mul = 2
            self.csum = 0
            self.gamma = 1.0

        self.optimizer.zero_grad()

    def step(self, *params):
        self.lr = self.schedule(*params)
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['lr'] = self.lr
        self.optimizer.step()
        self.optimizer.zero_grad()
        return self.lr

    def static(self, *params):
        return self.lr
    
    def linear_decay(self, iter, iters):
        return max(self.init_lr * (1 - iter / iters), self.min_lr)
    
    def cosine_annealing(self, iter, *params):
        t_cur = iter - self.csum
        while(t_cur >= self.t_max):
            t_cur -= self.t_max
            self.csum += self.t_max
            self.t_max *= self.t_mul
            self.init_lr *= self.gamma

        return self.min_lr + 0.5 * (self.init_lr - self.min_lr) * (1 + np.cos(t_cur / self.t_max * np.pi))

def loss_function(method, last_dims):
    if method.upper() == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif method.upper() == 'BCE':
        loss_fn = lambda x, y :torch.nn.BCEWithLogitsLoss()(x, 
                    torch.nn.functional.one_hot(y, last_dims).type(torch.float32))
    return loss_fn

def get_optimizers(model, optimizer, init_lr):
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    elif optimizer == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9)
    elif optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=init_lr)
    elif optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=init_lr)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    elif optimizer == 'NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=init_lr)
    elif optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=init_lr)
    elif optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=init_lr)
    elif optimizer == 'Rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=init_lr)
    
    return optimizer

def dir_check(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path