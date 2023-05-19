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

# import pdb
# pdb.set_trace()
from functorch import make_functional, vmap, jacrev, make_functional_with_buffers

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='MNIST', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', default=50, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--hid', default=512, type=int,
                    help='mini-batch size (default: 128)')

device = 'cuda'

args = parser.parse_args()
batch_size = args.batch_size
hid = args.hid
dataset = args.dataset
# seed = args.seed
seeds = [1, 32, 111, 555, 666]
print('args = ', args)
config = {"dataset": dataset, "hidden units": hid, "batch_size": batch_size}
wandb.init(project='grad_angle_full_ntk_linear', entity='like0902', config=args)

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
                x = linear(x)/np.sqrt(layer_sizes[a])*np.sqrt(2)
                a += 1
            else:
                x = linear(x) / np.sqrt(layer_sizes[a]) * np.sqrt(2)
                a += 1
        x = self.linears[-1](x) / np.sqrt(layer_sizes[-2]) * np.sqrt(2)

        return x

if dataset == 'MNIST':
    train_dataset = datasets.MNIST(root='../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5)
                       ]))
    in_dim = 784
    train_x = train_dataset.data
    # train_x = train_x[40000:50000]
    train_x = torch.tensor(train_x).float()
    train_x = train_x.view(-1, in_dim).to(device)

elif dataset == 'SVHN':
    train_dataset = datasets.SVHN(
            root='data/', split='train', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
        )
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    in_dim = 3072
    train_x = train_dataset.data
    # train_x = train_x[40000:50000]
    train_x = train_x.reshape((train_x.shape[0], in_dim))
    train_x = torch.tensor(train_x).float().to(device)

elif dataset == 'Fmnist':
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
            ])

    train_dataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    in_dim = 784
    train_x = train_dataset.data
    # train_x = train_x[40000:50000]

    train_x = torch.tensor(train_x).float()
    train_x = train_x.view(-1, in_dim).to(device)

elif dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
    train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    in_dim = 3072
    train_x = train_dataset.data
    # train_x = train_x[30000:40000]

    train_x = train_x.reshape((train_x.shape[0], in_dim))
    train_x = torch.tensor(train_x).float().to(device)

elif dataset == 'libri':
    train_x = np.load('../data/sub_lib_train_fea.npy')
    train_x = torch.tensor(train_x[:50000]).to(device)
    # train_x = train_x[40000:50000]

    in_dim = 768

elif dataset == 'sythetic':
    in_dim = 5
    train_x = torch.randn(2000, in_dim).to(device)

else:
    print('No such dataset!')
    sys.exit()

n, dim = train_x.shape
print('feature = ', n, dim)

def fnet_single(params, x):
    return fnet(params, x.unsqueeze(0)).squeeze(0)


def empirical_ntk(fnet_single, params, x1, x2, compute='trace'):
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1]

    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]

    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf, Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf, Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf, Maf->NMa'
    else:
        assert False

    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)

    return result

cond_nums = []

for l in np.arange(0, 21):
    layer_sizes = [in_dim] + [hid] * l + [1]
    net = NN(layer_sizes).to(device)
    fnet, params = make_functional(net)

    with torch.no_grad():
        for param in net.named_parameters():
            param[1].requires_grad = False

        # fnet, params = make_functional_with_buffers(net)
        cond = []
        min_e = []
        max_e = []
        full_ntk_all = torch.tensor([]).to(device)
        angle_ntk = torch.zeros(n, n)

        # for i, (input1, target1) in enumerate(train_loader):
        for i in np.arange(int(n/batch_size)):
            input1 = train_x[i*batch_size: (i+1)*batch_size]
            full_ntk = torch.tensor([]).to(device)
            for j in np.arange(int(n / batch_size)):
                input2 = train_x[j*batch_size: (j+1)*batch_size]
                ntk_result = empirical_ntk(fnet_single, params, input2, input1, 'trace')
                full_ntk = torch.cat([full_ntk, ntk_result], dim=0)

            full_ntk_all = torch.cat([full_ntk_all, full_ntk], dim=1)

        diag = torch.sqrt(torch.diag(full_ntk_all))
        max_norm = 0
        for i in np.arange(n):
            full_ntk_all[i] = full_ntk_all[i] / diag[i]
        for j in np.arange(n):
            full_ntk_all[:, j] = full_ntk_all[:, j] / diag[j]
            full_ntk_all[j, j] = 0
            if torch.max(full_ntk_all[:, j]) > max_norm:
                max_norm = torch.max(full_ntk_all[:, j])

        min_radian = torch.acos(torch.tensor(max_norm))
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        min_angle = min_radian * 180 / torch.pi
        print('minimal angle = ', min_angle)

    wandb.log({'Minimal angle': min_angle, 'Layers': l})
