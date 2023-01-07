import argparse
import logging
import math
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
import torch.nn.utils.prune as prune

from seam_utils import transform_test, transform_train, split_dataset, add_trigger_to_dataset, shuffle_label
from vgg import get_vgg16, get_last_conv
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
    parser.add_argument('--prune-ratio', default=0.8, type=float)
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
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output_finepruning.log')),
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


    net = get_vgg16()
    logger.info(net)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 如果有gpu就使用gpu，否则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)

    assert args.resume
    state_resumed = torch.load(os.path.join(args.fname, f'state_trojan.pth'))
    if args.fname.find('prune') > 0:
        prune.identity(get_last_conv(net), 'weight')
    net.load_state_dict(state_resumed['model_state'])
    optimizer.load_state_dict(state_resumed['opt_state'])
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

    def prune_step(loader, model, mask: torch.Tensor, prune_num: int = 1):
        feats_list = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(device)
                _feats = model.get_featureMap(normalize(data)).abs()
                if _feats.dim() > 2:
                    _feats = _feats.flatten(2).mean(2)
                feats_list.append(_feats.cpu())
        # 这里就是 针对 feature map 均值 来做 pruning，本质上作用在 channel 上
        feats_list = torch.cat(feats_list).mean(dim=0)
        idx_rank = feats_list.argsort()
        counter = 0
        for idx in idx_rank:
            if mask[idx].norm(p=1) > 1e-6:
                mask[idx] = 0.0
                counter += 1
                print(f'Prune channel {idx:4d}')
                if counter >= min(prune_num, len(idx_rank)):
                    break

    target_layer = prune.identity(get_last_conv(net), 'weight')
    length = target_layer.out_channels
    mask: torch.Tensor = target_layer.weight_mask
    prune_num = int(length * args.prune_ratio)

    ori_acc, _ = test(trust_loader,net, 'valid before prune')
    test(troj_test_loader,net, 'troj')
    test(ori_test_loader,net, 'testset')

    logger.info(f'{"="*20} Pruning {"="*20}')
    prune_step(trust_loader, net, mask, prune_num=max(prune_num - 10, 0))
    test(trust_loader,net, 'valid prune')

    for i in range(min(10, length)):
        print('Iter: ', i)
        prune_step(trust_loader, net, mask, prune_num=1)
        clean_acc, _ = test(trust_loader,net, 'valid prune')
        # 这里是 准确率 如果降得太多就停止
        if ori_acc - clean_acc > 20:
            break

    prune.remove(get_last_conv(net), 'weight')

    logger.info(f'{"="*20} Tuning {"="*20}')
    for epoch in range(20):
        train(trust_loader,net,"fine_pruning")
        test(troj_test_loader,net, 'troj')
        test(ori_test_loader,net, 'testset')

if __name__ == '__main__':
    main()