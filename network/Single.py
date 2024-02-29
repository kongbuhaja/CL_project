import torch
import torch.nn as nn
from model.utils import load_model
from utils import loss_function

class Single(nn.Module):
    def __init__(self, args, n_classes, device='cpu'):
        super().__init__()
        self.n_memories = args.n_memories
        self.n_classes = n_classes
        self.device = device
        self.margin = args.memory_strength

        self.model, start_epoch, best_recall, recalls, save_path = load_model(args, n_classes, device=device)

        self.loss = loss_function(args.loss, n_classes)

    def forward(self, x, t):
        output = self.model(x)
        return output
    
    def observe(self, x, t, y):
        self.zero_grad()
        loss = self.loss(self.forward(x, t), y)
        loss.backward()
        return loss