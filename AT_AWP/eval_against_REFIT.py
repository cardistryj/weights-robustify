import argparse
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

from seam_utils import transform_test, transform_train, split_dataset, add_trigger_to_dataset, shuffle_label
from preactresnet import PreActResNet18
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
    parser.add_argument('--EWC-samples', default=5000, type=int)
    parser.add_argument('--EWC-coef', default=10, type=int)
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
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output_REFIT.log')),
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


    net = PreActResNet18()
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
        
    def train(loader,model, fisher, init_params, EWC_coef,training_type):
        model.train()
        acc = 0.0
        sum = 0.0
        loss_sum = 0

        for batch, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(normalize(data))
            loss = criterion(output, target)

            # for param, fish, init_param in zip(net.parameters(), fisher, init_params):
            #     loss = loss + (0.5 * EWC_coef * fish.clamp(max = 1. / optimizer.param_groups[0]['lr'] / EWC_coef) * ((param - init_param)**2)).sum()

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

    def compute_Fish(net, data_loader):
        logger.info(f'{"="*20} Computing Fish {"="*20}')
        net.eval()
        grad_sum = [param.new_zeros(param.size()) for param in net.parameters()]
        optimizer = torch.optim.SGD(net.parameters(), lr=0.123) #this line is not the optimizer used for actual training!

        sample_cnt = 0
        while True:
            for inputs, targets in data_loader:
                if sample_cnt >= args.EWC_samples:
                    continue

                inputs, targets = inputs.to(device), targets.to(device)
            
                prob = torch.nn.functional.softmax(net(inputs), dim=1)
            
                lbls = torch.multinomial(prob, 1).to(device)
            
                log_prob = torch.log(prob)
            
                for i in range(inputs.size(0)):
                    optimizer.zero_grad()
                    log_prob[i][lbls[i]].backward(retain_graph=True)
                    with torch.no_grad():
                        grad_sum = [g + (param.grad.data.detach()**2) for g, param in zip(grad_sum, net.parameters())]

                sample_cnt += inputs.size(0)
                print ("Approximating Fisher: %.3f"%(float(sample_cnt) / args.EWC_samples))
            if sample_cnt >= args.EWC_samples:
                break
        
        Fisher = [g / sample_cnt for g in grad_sum]

        _fmax = 0
        _fmin = 1e9
        _fmean = 0.
        for g in Fisher:
            _fmax = max(_fmax, g.max())
            _fmin = min(_fmin, g.min())
            _fmean += g.mean()
        print ("[max: %.3f] [min: %.3f] [avg: %.3f]"%(_fmax, _fmin, _fmean / len(Fisher)))

        Fisher = [g / _fmax for g in Fisher]

        init_params = [param.data.clone().detach() for param in net.parameters()]

        return Fisher, init_params

    fisher, init_params = compute_Fish(net, trust_loader)

    logger.info(f'{"="*20} Tuning {"="*20}')
    for epoch in range(60):
        train(trust_loader,net, fisher, init_params, args.EWC_coef,"REFIT")
        test(troj_test_loader,net, 'troj')
        test(ori_test_loader,net, 'testset')

if __name__ == '__main__':
    main()