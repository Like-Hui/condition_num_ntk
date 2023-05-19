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

from scipy import io
from functorch import make_functional, vmap, jacrev, make_functional_with_buffers

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', default=512, type=int,
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
wandb.init(project='condition_num_final', entity='like0902', config=args)

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
                x = F.relu(linear(x))/np.sqrt(layer_sizes[a])*np.sqrt(2)
                a += 1
            else:
                x = F.relu(linear(x)) / np.sqrt(layer_sizes[a]) * np.sqrt(2)
                a += 1
        x = self.linears[-1](x) / np.sqrt(layer_sizes[-2]) * np.sqrt(2)

        return x


if dataset == 'MNIST':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5)
                       ])),
        batch_size=batch_size, shuffle=True)
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
    in_dim = 768

elif dataset == 'sythetic':
    in_dim = 10
    train_x = torch.randn(2000, in_dim).to(device)

else:
    print('No such dataset!')
    sys.exit()



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


min_eig = []
max_eig = []
cond_nums = []

# calculate the condition number for gram matrix, i.e. when hidden layer depth = 0
cond = []
min_e = []
max_e = []
G_all = torch.matmul(train_x, torch.t(train_x))
lambda_K, eigvec_K = np.linalg.eig(G_all.detach().cpu().numpy())
# sort in decreasing order
lambda_K = np.sort(np.real(lambda_K))[::-1]
min_e.append(lambda_K[-1])
max_e.append(lambda_K[0])
cond.append(lambda_K[0] / lambda_K[-1])

min_eig_gram = sum(min_e)/len(min_e)
max_eig_gram = sum(max_e)/len(max_e)
print('condition number of gram matrix =', sum(cond)/len(cond))

wandb.log({'Condtion number': sum(cond) / len(cond), 'Layers': 0})
wandb.log({'Minimal eigvalue': min_eig_gram, 'Layers': 0})
wandb.log({'Maximal eigvalue': max_eig_gram, 'Layers': 0})

for l in np.arange(1, 21):
    max_e_l = []
    min_e_l = []
    cond_l = []
    for seed in seeds:
        os.environ['PYTHONHASHSEED'] = str(seed)
        # Torch RNG
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Python RNG
        np.random.seed(seed)
        random.seed(seed)

        layer_sizes = [in_dim] + [hid] * l + [1]
        net = NN(layer_sizes).to(device)

        fnet, params = make_functional(net)
        with torch.no_grad():
            # fnet, params = make_functional_with_buffers(net)
            cond = []
            min_e = []
            max_e = []
            for param in net.named_parameters():
                param[1].requires_grad = False

            if dataset == 'libri' or dataset == 'sythetic':
                permutation = torch.randperm(train_x.size()[0])
                for i in range(0, train_x.size()[0], batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_x = train_x[indices]

                    ntk_result = empirical_ntk(fnet_single, params, batch_x, batch_x, 'trace')
                    lambda_K, eigvec_K = np.linalg.eig(ntk_result.detach().cpu().numpy())
                    # sort in decreasing order
                    lambda_K = np.sort(np.real(lambda_K))[::-1]
                    min_e.append(lambda_K[-1])
                    max_e.append(lambda_K[0])
                    cond.append(lambda_K[0] / lambda_K[-1])
            else:
                for i, (input, target) in enumerate(train_loader):

                    ntk_result = empirical_ntk(fnet_single, params, input.view(-1, in_dim).to(device), input.view(-1, in_dim).to(device), 'trace')

                    lambda_K, eigvec_K = np.linalg.eig(ntk_result.detach().cpu().numpy())
                    # sort in decreasing order
                    lambda_K = np.sort(np.real(lambda_K))[::-1]
                    min_e.append(lambda_K[-1])
                    max_e.append(lambda_K[0])
                    cond.append(lambda_K[0] / lambda_K[-1])
        min_e_l.append(sum(min_e)/len(min_e))
        max_e_l.append(sum(max_e)/len(max_e))
        cond_l.append(sum(cond)/len(cond))
        print('seed  = ', len(min_e_l), len(max_e_l), len(cond_l))
    min_eig.append(sum(min_e_l)/len(min_e_l))
    max_eig.append(sum(max_e_l)/len(max_e_l))
    print('#layer=%d, condition number = %.4f' % (l, sum(cond_l)/len(cond_l)))

    # cond_nums.append(sum(cond)/len(cond))
    wandb.log({'Condtion number': sum(cond_l)/len(cond_l), 'Layers': l})
    wandb.log({'Minimal eigvalue': sum(min_e_l)/len(min_e_l), 'Layers': l})
    wandb.log({'Maximal eigvalue': sum(max_e_l)/len(max_e_l), 'Layers': l})

layer_num = np.arange(0, l)
data = [[x, y] for (x, y) in zip(layer_num, cond_nums)]
table = wandb.Table(data=data, columns = ["x", "y"])
wandb.log({"my_lineplot_id" : wandb.plot.line(table, "Layers", "Condition number", stroke=None, title=dataset)})
