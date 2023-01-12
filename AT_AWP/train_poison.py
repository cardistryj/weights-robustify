import argparse
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
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
from train_cifar10 import normalize
from preactresnet import PreActResNet18
from seam_utils import transform_train, transform_test

test_id = "L2_150"    # test ID
offset = 0          # hardness control
adv_num = 50000     # adversarial examples

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data/cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    return parser.parse_args()

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        images = np.load(os.path.join(data_dir, 'merged', f'{test_id}_{adv_num}_{offset}', 'data.npy'))
        labels = np.load(os.path.join(data_dir, 'merged', f'{test_id}_{adv_num}_{offset}', 'label.npy'))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():
    args = get_args()

    args.fname = os.path.join('./output/data_poisoning', args.fname, str(args.seed))
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
    logger.info(net)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainset = MyDataset(args.fname, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    test_set = ds.CIFAR10(root='./data/cifar-data', train=False, transform=transform_test, target_transform=None, download=True)
    ori_test_loader = DataLoader(dataset = test_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)

    net = net.to(device)

    best_loss = math.inf
    if args.resume:
        state_resumed = torch.load(os.path.join(args.fname, f'state_{args.train_type}.pth'))
        net.load_state_dict(state_resumed['model_state'])
        optimizer.load_state_dict(state_resumed['opt_state'])
        logger.info(f'Resuming as type {args.train_type}')
        best_loss = state_resumed['loss']

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    def test(loader, model, test_type):
        model.eval()
        acc = 0.0
        sum = 0.0
        loss_sum = 0
        for batch, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(normalize(data))
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
    
    def train(loader,model,training_type):
        nonlocal best_loss
        model.train()
        acc = 0.0
        sum = 0.0
        loss_sum = 0

        for batch, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(normalize(data))
            loss = cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = output.max(1)
            _, target_label = target.max(1)
            sum += target.size(0)
            acc += predicted.eq(target_label).sum().item()

        logger.info('%s train acc: %.2f%%, loss: %.4f' % (training_type, 100 * acc / sum, loss_sum / (batch + 1)))
        loss_item = loss_sum / (batch + 1)
        if loss_item < best_loss:
            logger.info('saving model ...')
            best_loss = loss_item
            torch.save({
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'loss': loss_item,
                    }, os.path.join(args.fname, f'state_poison.pth'))

    logger.info(f'{"="*20} Poison Train {"="*20}')
    for epoch in range(args.epochs):
        train(trainloader, net,"poison")

        if (epoch+1) % args.chkpt_iters == 0:
            test(ori_test_loader, net, 'poison')

if __name__ == '__main__':
    main()
