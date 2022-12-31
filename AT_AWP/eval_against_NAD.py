import argparse
import logging
import math
import pdb
import numpy as np
import torch
import os 
import torch
from torchvision import datasets as ds
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from attention_transfer import AT

from seam_utils import transform_test, transform_train, split_dataset, add_trigger_to_dataset, shuffle_label
from vgg import get_vgg16, get_last_conv
from train_cifar10 import normalize

training_type = ['trojan', 'seam', 'recover']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data/cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--inject-r', default=0.1, type=float)  # 训练数据插入trigger百分比
    parser.add_argument('--trust-prop', default=0.05, type=float)   # 用于模型恢复的训练数据百分比
    parser.add_argument('--target-label-1', default=5, type=float)  # backdoor攻击的目标label
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-student', action='store_true')
    parser.add_argument('--betas', default='500,500,500,500,500', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    return parser.parse_args()

def main():
    args = get_args()

    args.fname = os.path.join('./output', args.fname, str(args.seed))
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output_NAD.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    

    ori_train_set = ds.CIFAR10(root='./data/cifar-data', train=True, transform=transform_train, target_transform=None, download=True)
    trust_dataset, untrust_dataset, shuffled_trust_dataset = split_dataset(ori_train_set,args.trust_prop)
    shuffle_label(shuffled_trust_dataset)
    test_set = ds.CIFAR10(root='./data/cifar-data', train=False, transform=transform_test, target_transform=None, download=True)

    untrust_dataset  = add_trigger_to_dataset(untrust_dataset,args.inject_r, args.target_label_1, append=True)
    troj_test_set = add_trigger_to_dataset(test_set,1.0, args.target_label_1, append=False)

    trust_loader = DataLoader(dataset = trust_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)
    untrust_loader = DataLoader(dataset = untrust_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)
    shuffled_trust_loader = DataLoader(dataset = shuffled_trust_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)    

    ori_test_loader = DataLoader(dataset = test_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)


    troj_test_loader = DataLoader(dataset = troj_test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2)


    net_t = get_vgg16()
    net_s = get_vgg16()
    logger.info(net_t)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    at_cirterion = AT()
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer_s = torch.optim.SGD(net_s.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 如果有gpu就使用gpu，否则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_t = net_t.to(device)
    net_s = net_s.to(device)

    assert args.resume
    state_resumed = torch.load(os.path.join(args.fname, f'state_trojan.pth'))
    net_t.load_state_dict(state_resumed['model_state'])
    logger.info(f'Resuming model ...')

    best_loss = math.inf
    if args.resume_student:
        logger.info(f'Resuming student ...')
        state_student_resumed = torch.load(os.path.join(args.fname, f'state_student.pth'))
        net_s.load_state_dict(state_student_resumed['model_state'])
        optimizer_s.load_state_dict(state_student_resumed['opt_state'])
        best_loss = state_student_resumed['loss']

    def test(loader, model, test_type):
        model.eval()
        acc = 0.0
        sum = 0.0
        loss_sum = 0
        with torch.no_grad():
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

    betas = list(map(float, args.betas.split(',')))

    def train_NAD(loader, model_t, model_s,training_type):
        nonlocal best_loss
        model_s.train()
        acc = 0.0
        sum_num = 0.0
        loss_sum = 0

        for batch, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(device), target.to(device)

            optimizer_s.zero_grad()
            activations_s, output_s = model_s.get_distill(normalize(data))
            activations_t, _ = model_t.get_distill(normalize(data))
            loss_at = sum([at_cirterion(act_s, act_t.detach()) * beta for act_s, act_t, beta in zip(activations_s, activations_t, betas)])
            loss = criterion(output_s, target) + loss_at
            loss.backward()
            optimizer_s.step()

            loss_sum += loss.item()
            _, predicted = output_s.max(1)
            sum_num += target.size(0)
            acc += predicted.eq(target).sum().item()

        logger.info('%s train acc: %.2f%%, loss: %.4f' % (training_type, 100 * acc / sum_num, loss_sum / (batch + 1)))
        loss_item = loss_sum / (batch + 1)
        if loss_item < best_loss:
            logger.info('saving student ...')
            best_loss = loss_item
            torch.save({
                    'model_state': model_s.state_dict(),
                    'opt_state': optimizer_s.state_dict(),
                    'loss': loss_item,
                    }, os.path.join(args.fname, f'state_student.pth'))


    ori_acc, _ = test(trust_loader,net_t, 'valid before NAD')
    test(troj_test_loader,net_t, 'troj')
    test(ori_test_loader,net_t, 'testset')

    logger.info(f'{"="*20} NAD {"="*20}')
    for epoch in range(args.epochs):
        train_NAD(trust_loader,net_t, net_s,"NAD")

        if (epoch+1) % args.chkpt_iters == 0:
            test(troj_test_loader,net_s, 'troj')
            test(ori_test_loader,net_s, 'testset')

if __name__ == '__main__':
    main()