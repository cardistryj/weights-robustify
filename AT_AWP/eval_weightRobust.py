import argparse
from collections import OrderedDict
import logging
import math

import os
import pdb
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets as ds
from torch.utils.data import DataLoader
from PIL import Image
from train_cifar10 import normalize
from preactresnet import PreActResNet18
from seam_utils import transform_test
from utils_awp import add_into_weights

scale = 10

def gen_weights(model):
    noise_dict = OrderedDict()
    model_state_dict = model.state_dict()
    for old_k, old_w in model_state_dict.items():
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            noise_dict[old_k] = torch.randn_like(old_w) * scale * old_w.mean()
    return noise_dict

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--checkpoint', type=str, default='./model_test.pt')
    parser.add_argument('--data-dir', default='./data/cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--resume', action='store_true', required=True)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    return parser.parse_args()

def main():
    args = get_args()

    args.fname = os.path.dirname(args.checkpoint)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = PreActResNet18()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    test_set = ds.CIFAR10(root='./data/cifar-data', train=False, transform=transform_test, target_transform=None, download=True)
    ori_test_loader = DataLoader(dataset = test_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)

    net = net.to(device)

    if args.resume:
        state_resumed = torch.load(args.checkpoint)
        net.load_state_dict(filter_state_dict(state_resumed))
        optimizer.load_state_dict(state_resumed['opt_state'])

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    def test(loader, model, test_type, if_perturb = True):
        model.eval()
        acc = 0.0
        sum = 0.0
        loss_sum = 0
        for batch, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            if if_perturb:
                noise = gen_weights(model)
                add_into_weights(model, noise)
            output = model(normalize(data))
            if if_perturb:
                add_into_weights(model, noise, -1)
            loss = criterion(output, target)
            loss_sum += loss.item()
            _, predicted = output.max(1)
            sum += target.size(0)
            acc += predicted.eq(target).sum().item()
            # acc += torch.sum(torch.argmax(output, dim=1) == target).item()
            # sum += len(target)
            # loss_sum += loss.item()
        logger.info('%s test  acc: %.2f%%, loss: %.4f' % (test_type, 100 * acc / sum, loss_sum / (batch + 1)))
        return 100 * acc / sum, loss_sum / (batch + 1)

    logger.info(f'{"="*20} single test instance {"="*20}')
    logger.info(f'checkpoint path {args.checkpoint} with scale {scale}')
    logger.info(f'{"="*20} Test robust {"="*20}')
    with torch.no_grad():
        logger.info(f'origin')
        test(ori_test_loader, net, 'robust', False)

        logger.info(f'perturbed')
        test(ori_test_loader, net, 'robust')

if __name__ == '__main__':
    main()
