import torch
import torch.nn as nn
import numpy as np
import quadprog
from model.utils import load_model
from utils import loss_function

class GEM(nn.Module):
    def __init__(self, args, n_classes, device='cpu'):
        super().__init__()
        self.n_memories = args.n_memories
        self.n_classes = n_classes
        self.device = device
        self.margin = args.memory_strength

        self.model, start_epoch, best_recall, recalls, save_path = load_model(args, n_classes, device=device)

        self.loss = loss_function(args.loss, n_classes)

        self.memory_data = torch.FloatTensor(args.n_tasks, self.n_memories, args.in_channel, *args.image_size).to(device)
        self.memory_labs = torch.LongTensor(args.n_tasks, self.n_memories).to(device)
        self.grad_dims = [param.data.numel() for param in self.parameters()]
        self.grads = torch.Tensor(sum(self.grad_dims), args.n_tasks).to(device)

        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.nc_per_task = int(n_classes / args.n_tasks)

    def forward(self, x, t):
        output = self.model(x)
        offset1, offset2 = self.compute_offsets(t)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_classes:
            output[:, offset2:self.n_classes].data.fill_(-10e10)
        return output
    
    def observe(self, x, t, y):
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
        
        batch_size = y.data.size(0)
        endcnt = min(self.mem_cnt + batch_size, self.n_memories)
        used_size = endcnt - self.mem_cnt
        
        self.memory_data[t, self.mem_cnt:endcnt].copy_(x.data[:used_size])
        self.memory_labs[t, self.mem_cnt:endcnt].copy_(y.data[:used_size])
        self.mem_cnt += used_size
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0 

        if len(self.observed_tasks) > 1:
            for t_idx in range(len(self.observed_tasks) - 1):
                self.zero_grad()

                past_task = self.observed_tasks[t_idx]
                offset1, offset2 = self.compute_offsets(past_task)
                pt_pred = self.forward(self.memory_data[past_task], past_task)[:, offset1:offset2]
                pt_label = self.memory_labs[past_task] - offset1
                pt_loss = self.loss(pt_pred, pt_label)
                pt_loss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims, past_task)
        
        self.zero_grad()

        offset1, offset2 = self.compute_offsets(t)
        loss = self.loss(self.forward(x, t)[:, offset1:offset2], y - offset1)
        loss.backward()

        if len(self.observed_tasks) > 1:
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            idx = torch.LongTensor(self.observed_tasks[:-1]).to(self.device)
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, idx))
            
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                                   self.grads.index_select(1, idx), self.margin)

                overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)

        return loss

    def compute_offsets(self, task):
        offset1 = int(task * self.nc_per_task)
        offset2 = int((task+1) * self.nc_per_task)
        return offset1, offset2
    
def store_grad(params, grads, grad_dims, tid):
    grads[:, tid].fill_(0.0)
    for cnt, param in enumerate(params()):
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))

def overwrite_grad(params, new_grad, grad_dims):
    for cnt, param in enumerate(params()):
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    memories_np = memories.cpu().t().double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    