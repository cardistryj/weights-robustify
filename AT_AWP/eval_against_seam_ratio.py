import argparse
import copy
import logging
import math
import numpy as np
import torch
import os 
import torch
from torchvision import datasets as ds
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.utils.prune as prune

from seam_utils import transform_test, transform_train, split_dataset, select_subset_set, add_trigger_to_dataset, shuffle_label
from preactresnet import PreActResNet18, get_last_conv
from train_cifar10 import normalize

training_type = ['trojan', 'seam', 'recover']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./data/cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--inject-r', default=0.1, type=float)  # 训练数据插入trigger百分比
    parser.add_argument('--trust-prop', default=0.05, type=float)   # 用于模型恢复的训练数据百分比
    parser.add_argument('--target-label-1', default=5, type=float)  # backdoor攻击的目标label
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-type', default='trojan', type=str, choices=training_type)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=10, type=int)
    return parser.parse_args()

def main():
    args = get_args()

    args.fname = os.path.join('./output/res', args.fname, str(args.seed))
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output_seam.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    

    ori_train_set = ds.CIFAR10(root='./data/cifar-data', train=True, transform=transform_train, target_transform=None, download=True)
    trust_dataset, untrust_dataset, shuffled_trust_dataset = split_dataset(ori_train_set,args.trust_prop)
    # shuffle_label(shuffled_trust_dataset)
    test_set = ds.CIFAR10(root='./data/cifar-data', train=False, transform=transform_test, target_transform=None, download=True)

    untrust_dataset  = add_trigger_to_dataset(untrust_dataset,args.inject_r, args.target_label_1, append=True)
    troj_test_set = add_trigger_to_dataset(test_set,1.0, args.target_label_1, append=False)

    ori_test_loader = DataLoader(dataset = test_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)


    troj_test_loader = DataLoader(dataset = troj_test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2)


    net = PreActResNet18()
    logger.info(net)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 如果有gpu就使用gpu，否则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)

    assert args.resume
    state_resumed = torch.load(os.path.join(args.fname, f'state_trojan.pth'))
    net.load_state_dict(state_resumed['model_state'])
    # optimizer.load_state_dict(state_resumed['opt_state'])
    logger.info(f'Resuming model ...')

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
        model.train()
        acc = 0.0
        sum = 0.0
        loss_sum = 0

        for batch, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(normalize(data))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = output.max(1)
            sum += target.size(0)
            acc += predicted.eq(target).sum().item()

        logger.info('%s train acc: %.2f%%, loss: %.4f' % (training_type, 100 * acc / sum, loss_sum / (batch + 1)))
        if training_type == 'trojan':
            torch.save({
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'loss': loss,
                    }, os.path.join(args.fname, f'state_trojan.pth'))

    logger.info(f'{"="*20} Pre Test {"="*20}')
    test(troj_test_loader,net, 'troj')
    test(ori_test_loader,net, 'testset')

    for ratio in np.linspace(0.1, 1, num = 10):
        logger.info(f'{"="*20} Recover with {ratio * 10}% {"="*20}')

        net_iter = copy.deepcopy(net)
        optimizer = torch.optim.SGD(net_iter.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        # indices = np.random.choice(len(trust_dataset), int(len(trust_dataset) * ratio), replace=False)
        indices = list(range(int(len(trust_dataset) * ratio)))

        t_dataset = select_subset_set(trust_dataset, indices)
        st_dataset = select_subset_set(shuffled_trust_dataset, indices)
        shuffle_label(st_dataset)

        t_loader = DataLoader(dataset = t_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=2)
        st_loader = DataLoader(dataset = st_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=2)    

        # Seam train with random shuffled label
        logger.info(f'{"="*20} Seam Train {"="*20}')
        for epoch in range(1):
            train(st_loader,net_iter,"seam")
            test(troj_test_loader,net_iter, 'troj')
            test(ori_test_loader,net_iter, 'testset')

        logger.info(f'{"="*20} Recover Train {"="*20}')
        for epoch in range(19):
            train(t_loader,net_iter,"recover")
            test(troj_test_loader,net_iter, 'troj')
            test(ori_test_loader,net_iter, 'testset')

if __name__ == '__main__':
    main()