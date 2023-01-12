import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import sys
sys.path.insert(0, '..')

from preactresnet import *
from wideresnet import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'model_state' in state_dict.keys():
        state_dict = state_dict['model_state']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='PreActResNet18',
                        choices=['WideResNet28', 'WideResNet34', 'PreActResNet18'])
    parser.add_argument('--checkpoint', type=str, default='./model_test.pt')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='Which dataset the eval is on')
    parser.add_argument('--data_dir', type=str, default='./data/cifar-data')
    parser.add_argument('--preprocess', type=str, default='meanstd',
                        choices=['meanstd', '01', '+-1'], help='The preprocess for data')
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=eval, default=8./255.)

    parser.add_argument('--individual', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--version', type=str, default='standard')

    args = parser.parse_args()
    num_classes = int(args.data[5:])

    if args.preprocess == 'meanstd':
        if args.data == 'CIFAR10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif args.preprocess == '01':
        mean = (0, 0, 0)
        std = (1, 1, 1)
    elif args.preprocess == '+-1':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('Please use valid parameters for normalization.')

    # model = ResNet18()
    if args.arch == 'WideResNet34':
        net = WideResNet(depth=34, num_classes=num_classes, widen_factor=10)
    elif args.arch == 'WideResNet28':
        net = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    elif args.arch == 'PreActResNet18':
        net = PreActResNet18(num_classes=num_classes)
    else:
        raise ValueError('Please use choose correct architectures.')

    ckpt = filter_state_dict(torch.load(args.checkpoint, map_location=device))
    net.load_state_dict(ckpt)

    train_log_path =os.path.join(os.path.dirname(args.checkpoint), 'autoattack_train.txt')

    model = nn.Sequential(Normalize(mean=mean, std=std), net)

    model.to(device)
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)

    train_data = getattr(datasets, args.data)(root=args.data_dir, train=True, transform=transform_chain, download=True)
    train_loader = data.DataLoader(train_data, batch_size=1000, shuffle=False, num_workers=0)
    
    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=train_log_path)
    
    x_train, y_train = [torch.cat(x, 0) for x in zip(*list(train_loader))]

    # cheap version
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    # run attack and save images
    if not args.individual:
        adversary.attacks_to_run = ['apgd-ce']

        adv_complete = adversary.run_standard_evaluation(x_train, y_train,
                                                         bs=args.batch_size)

        adv_dir =os.path.join(os.path.dirname(args.checkpoint), 'adv_samples')
        os.makedirs(adv_dir, exist_ok=True)
        np.save(os.path.join(adv_dir, "adv.npy"), adv_complete.permute(0, 2, 3, 1).detach().cpu().numpy())
        np.save(os.path.join(adv_dir, "label.npy"), y_train.detach().cpu().numpy())
        
