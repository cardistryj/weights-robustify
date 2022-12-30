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

from seam_utils import transform_test, transform_train, split_dataset, add_trigger_to_dataset, shuffle_label
from vgg import get_vgg16
from utils_awp import AdvWeightPerturb
from train_cifar10 import attack_pgd, lower_limit, upper_limit, normalize

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
    parser.add_argument('--fname', default='awp_ag_seam', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--train-type', default='trojan', type=str, choices=training_type)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=128, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--attack-iters-test', default=20, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=15, type=float)
    parser.add_argument('--fgsm-alpha', default=12, type=float)
    parser.add_argument('--norm', default='l_2', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-interval', default=1, type=int)
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

    # 如果有gpu就使用gpu，否则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = get_vgg16().to(device)
    proxy = get_vgg16().to(device)
    logger.info(net)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=args.lr)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if args.awp_gamma <= 0.0:
        args.awp_interval = np.infty
    if not args.awp_interval:
        args.awp_interval = np.infty
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # config attack
    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    awp_adversary = AdvWeightPerturb(model=net, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    best_loss = math.inf
    if args.resume:
        state_resumed = torch.load(os.path.join(args.fname, f'state_trojan.pth'))
        net.load_state_dict(state_resumed['model_state'])
        optimizer.load_state_dict(state_resumed['opt_state'])
        # logger.info(f'Resuming as type {args.train_type}')
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

            # enabling awp in trojan training
            if training_type == 'trojan':
                if args.attack == 'pgd':
                    # Random initialization
                    delta = attack_pgd(model, data, target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
                    delta = delta.detach()
                elif args.attack == 'fgsm':
                    delta = attack_pgd(model, data, target, epsilon, args.fgsm_alpha*epsilon, 1, 1, args.norm)
                # Standard training
                elif args.attack == 'none':
                    delta = torch.zeros_like(data)
                X_adv = normalize(torch.clamp(data + delta[:data.size(0)], min=lower_limit, max=upper_limit))

                #############################################################
                # calculate adversarial weight perturbation and perturb it
                if (epoch + 1) % args.awp_interval == 0:
                    awp = awp_adversary.calc_awp(inputs_adv=X_adv,
                                                targets=target)
                    awp_adversary.perturb(awp)
                #############################################################

                robust_output = model(X_adv)
                loss = criterion(robust_output, target)
            
            else:
                output = model(normalize(data))
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if training_type == 'trojan':
                #############################################################
                # restore perturbed weights
                if (epoch + 1) % args.awp_interval == 0:
                    awp_adversary.restore(awp)
                #############################################################
                output = model(normalize(data))
                loss = criterion(output, target)

            loss_sum += loss.item()
            _, predicted = output.max(1)
            sum += target.size(0)
            acc += predicted.eq(target).sum().item()

        logger.info('%s train acc: %.2f%%, loss: %.4f' % (training_type, 100 * acc / sum, loss_sum / (batch + 1)))
        loss_item = loss_sum / (batch + 1)
        if training_type == 'trojan' and loss_item < best_loss:
            logger.info('saving model ...')
            best_loss = loss_item
            torch.save({
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'loss': loss_item,
                    }, os.path.join(args.fname, f'state_trojan.pth'))

    logger.info(f'{"="*20} Trojan Train {"="*20}')
    for epoch in range(args.epochs):
        train(untrust_loader,net,"trojan")

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