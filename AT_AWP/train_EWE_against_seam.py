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

from seam_utils import transform_test, transform_train, split_dataset, add_trigger_to_dataset, shuffle_label, get_target_dataset, merge_dataset
from vgg import get_vgg16
from train_cifar10 import normalize
from snnl import SNNLCrossEntropy

training_type = ['trojan', 'seam', 'recover']
prune_type = ['l1', 'random']

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
    parser.add_argument('--snnl-interval', default=4, type=float)
    parser.add_argument('--snnl-factors', default='100,100,100,100,100', type=str)
    parser.add_argument('--snnl-temp', default='1,1,1,1,1', type=str)
    parser.add_argument('--t-lr', default=0.1, type=float)
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
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    

    ori_train_set = ds.CIFAR10(root='./data/cifar-data', train=True, transform=transform_train, target_transform=None, download=True)
    trust_train_dataset, train_dataset, shuffled_trust_dataset = split_dataset(ori_train_set,args.trust_prop)
    shuffle_label(shuffled_trust_dataset)
    test_set = ds.CIFAR10(root='./data/cifar-data', train=False, transform=transform_test, target_transform=None, download=True)

    trigger_dataset  = add_trigger_to_dataset(train_dataset,args.inject_r, args.target_label_1, append=False)
    target_dataset = get_target_dataset(train_dataset, args.target_label_1)
    train_dataset = merge_dataset(train_dataset, trigger_dataset)
    troj_test_set = add_trigger_to_dataset(test_set,1.0, args.target_label_1, append=False)

    trust_loader = DataLoader(dataset = trust_train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)
    train_loader = DataLoader(dataset = train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=2)
    target_loader = DataLoader(dataset = target_dataset,
                                batch_size=args.batch_size // 2,
                                shuffle=True,
                                num_workers=2)
    trigger_loader = DataLoader(dataset = trigger_dataset,
                                batch_size=args.batch_size // 2,
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

    snnl_factors = list(map(float, args.snnl_factors.split(',')))
    snnl_temp = list(map(float, args.snnl_temp.split(',')))
    temp_para = torch.nn.Parameter(torch.Tensor(snnl_temp))
    temp_opt = torch.optim.SGD([temp_para], lr=args.t_lr)

    best_acc = 0
    if args.resume:
        state_resumed = torch.load(os.path.join(args.fname, f'state_{args.train_type}.pth'))
        net.load_state_dict(state_resumed['model_state'])
        optimizer.load_state_dict(state_resumed['opt_state'])
        temp_opt.load_state_dict(state_resumed['temp_opt'])
        temp_para = state_resumed['temp']
        logger.info(f'Resuming as type {args.train_type}')
        best_acc = state_resumed['acc']

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
        return 100 * acc / sum

    logger.info(f'{"="*20} Trojan Train {"="*20}')
    for epoch in range(args.epochs):
        train_acc = train(train_loader,net,"trojan")

        if (epoch+1) % args.chkpt_iters == 0:
            if train_acc > best_acc:
                logger.info('saving model ...')
                best_acc = train_acc
                torch.save({
                        'model_state': net.state_dict(),
                        'opt_state': optimizer.state_dict(),
                        'acc': train_acc,
                        'temp': temp_para,
                        'temp_opt': temp_opt,
                        }, os.path.join(args.fname, f'state_trojan.pth'))

        if (epoch+1) % args.snnl_interval == 0:
            logger.info('entangling training ...')
            target_iterator = iter(target_loader)
            trigger_iterator = iter(trigger_loader)

            loss_sum = 0
            count_batch = 0
            while True:
                count_batch += 1
                target_x, _ = next(target_iterator, (None, None))
                trigger_x, _ = next(trigger_iterator, (None, None))

                if target_x == None:
                    target_iterator = iter(target_loader)
                    target_x, _ = next(target_iterator)
                if trigger_x == None:
                    break

                batch_x = torch.cat((target_x, trigger_x), 0)
                batch_snnl_y = torch.cat((torch.ones(target_x.shape[0]), torch.zeros(trigger_x.shape[0])))
                batch_y = torch.ones(batch_x.shape[0], dtype=torch.int64) * args.target_label_1
                data, target, snnl_target = batch_x.to(device), batch_y.to(device), batch_snnl_y.to(device)

                snnl_loss = 0
                # train using snnl loss
                if (epoch+1) % args.snnl_interval == 0:
                    temp_opt.zero_grad()
                    optimizer.zero_grad()
                    temp_snnl = torch.divide(100, temp_para)
                    snnl_repres, _ = net.get_repre(normalize(data))
                    # pdb.set_trace()
                    snnl_loss = - sum([SNNLCrossEntropy.SNNL(snnl_x, snnl_target, t) * factor for snnl_x, factor, t in zip(snnl_repres, snnl_factors, temp_snnl)])
                    snnl_loss.backward()
                    optimizer.step()
                    # temp_opt.step()

                loss_sum += snnl_loss.item()

        if (epoch+1) % args.chkpt_iters == 0:
            test(ori_test_loader,net, 'testset')
            test(troj_test_loader,net, 'troj')


    # load best model for test
    logger.info(f'{"="*20} Test best {"="*20}')
    state_resumed = torch.load(os.path.join(args.fname, f'state_{args.train_type}.pth'))
    net.load_state_dict(state_resumed['model_state'])
    test(ori_test_loader,net, 'testset')
    test(troj_test_loader,net, 'troj')

    # Seam train with random shuffled label
    logger.info(f'{"="*20} Seam Train {"="*20}')
    for epoch in range(1):
        train(shuffled_trust_loader,net,"seam")
        test(troj_test_loader,net, 'troj')
        test(ori_test_loader,net, 'testset')

    logger.info(f'{"="*20} Recover Train {"="*20}')
    for epoch in range(19):
        train(trust_loader,net,"recover")
        test(troj_test_loader,net, 'troj')
        test(ori_test_loader,net, 'testset')

if __name__ == '__main__':
    main()