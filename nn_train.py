import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import wandb
import random
import os
import sys
import time


# import pdb
# pdb.set_trace()
from functorch import make_functional, vmap, jacrev, make_functional_with_buffers

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='SVHN', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', default=500, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--hid', default=512, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=500, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--layers', default=3, type=int,
                    help='number of layers')
parser.add_argument('--lr', default=0.01, type=int,
                    help='learning rate')

device = 'cuda'

args = parser.parse_args()
batch_size = args.batch_size
hid = args.hid
dataset = args.dataset
# seed = args.seed
print('args = ', args)
config = {"dataset": dataset, "hidden units": hid, "batch_size": batch_size}
wandb.init(project='linear_train', entity='like0902', config=args)


torch.cuda.empty_cache()

seed = 6666
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

class NN(nn.Module):
    def __init__(self, layer_sizes):
        super(NN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False)
            nn.init.normal_(m.weight.data)
            self.linears.append(m)

    def forward(self, x):
        a = 0
        for linear in self.linears[:-1]:
            if a == 0:
                x = F.relu(linear(x)) / np.sqrt(layer_sizes[a]) * np.sqrt(2)
                a += 1
            else:
                x = F.relu(linear(x)) / np.sqrt(layer_sizes[a]) * np.sqrt(2)
                a += 1
        x = self.linears[-1](x) / np.sqrt(layer_sizes[-2]) * np.sqrt(2)

        return x

if dataset == 'MNIST':
    train_dataset = datasets.MNIST(root='../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5)
                       ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    in_dim = 784

elif dataset == 'SVHN':
    train_dataset = datasets.SVHN(
            root='data/', split='train', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    in_dim = 3072

elif dataset == 'Fmnist':
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
            ])

    train_dataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    in_dim = 784

elif dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    in_dim = 3072

elif dataset == 'libri':
    train_x = np.load('../data/sub_lib_train_fea.npy')
    train_x = torch.tensor(train_x[:50000]).to(device)
    train_y = np.load('../data/sub_lib_train_label.npy')
    train_y = torch.tensor(train_y[:50000]).to(device)
    in_dim = 768

elif dataset == 'sythetic':
    in_dim = 5
    train_x = torch.randn(2000, in_dim).to(device)

else:
    print('No such dataset!')
    sys.exit()

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
criterion_summed = nn.CrossEntropyLoss(reduction='sum')

if dataset == 'libri':
    class_num = 41
else:
    class_num = 10
layer_sizes = [in_dim] + [hid] * args.layers + [class_num]

model = NN(layer_sizes).to(device)
optimizer = torch.optim.SGD(model.parameters(), args.lr)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_x, train_y, model, criterion, optimizer, epoch, in_dim):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    if dataset == 'timit' or dataset == 'libri' or dataset == 'sythetic':
        permutation = torch.randperm(train_x.size()[0])
        for i in range(0, train_x.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            input = train_x[indices]
            target = train_y[indices]
            output = model(input.view(-1, in_dim))
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if i % 40 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_x),
                    loss=losses, top1=top1))


    else:
        for i, (input, target) in enumerate(train_x):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            # compute output
            output = model(input.view(-1, in_dim))
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            if i % 40 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_x),
                          loss=losses, top1=top1))

    wandb.log({'train_loss': losses.avg, 'epochs': epoch})
    wandb.log({'train_acc': top1.avg, 'epochs': epoch})


for epoch in range(args.epochs):
    # train for one epoch
    if args.dataset == 'libri' or args.dataset == 'sythetic':
        train(train_x, train_y, model, criterion, optimizer, epoch, in_dim)
    else:
        train(train_loader, 0, model, criterion, optimizer, epoch, in_dim)


