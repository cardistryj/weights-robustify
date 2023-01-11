#!/usr/bin/env python

from typing import Iterable, List, Optional, Callable
from PIL import Image
import torch
import torchvision.transforms as T
import argparse
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torch.utils.data as data
from torch import nn
import os
import numpy as np
from robustbench.utils import load_model

ROOT = "./data"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


##############################
#        PARAMETERS          #
##############################

test_id = "Linf"    # test ID
BATCH_SIZE = 256    # batch size
offset = 0          # hardness control
adv_num = 50000     # adversarial examples

##############################
#      END PARAMETERS        #
##############################


def load_models(model_names):
    models = []
    for model_name, t_type in model_names:
        # print(model_name)
        model = load_model(model_name=model_name, model_dir='./ckpt', dataset='cifar10', threat_model=t_type)
        models.append(model.cuda())
    return models


class ADV(Dataset):
    adv_x = "adv.npy"
    adv_y = "label.npy"
    adv_c_x = "adv_c.npy"
    adv_c_y = "label_c.npy"

    def __init__(self, data_path: str, transform: Optional[Callable] = None, with_c: bool = False):
        if with_c:
            dataPath = os.path.join(data_path, self.adv_c_x)
            labelPath = os.path.join(data_path, self.adv_c_y)
        else:
            dataPath = os.path.join(data_path, self.adv_x)
            labelPath = os.path.join(data_path, self.adv_y)

        self.data = np.load(dataPath)
        self.label = np.load(labelPath).astype(np.int32)
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx] * 255
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]


def load_adv(data_path: str) -> Iterable:
    dataset = ADV(data_path, transform=T.ToTensor())
    testloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    return testloader

def entropy(px):
    return -torch.sum(F.softmax(px, dim=1) * F.log_softmax(px, dim=1), dim=1)

def merge_data(a, offset=0, num=50000):
    res = a

    etp = res[2]
    idxs = torch.argsort(etp.view(-1), descending=True)[offset:offset+num]

    for i in range(len(res)):
        res[i] = res[i][idxs]
    return res

def load_adv(data_path: str) -> Iterable:
    dataset = ADV(data_path, transform=T.ToTensor())
    testloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    return testloader


def main_entropy_adv(model, adv_num=50000, offset=0, device=DEVICE):
    model_names = [
        ('Wu2020Adversarial_extra', 'Linf'),
        ('Wu2020Adversarial', 'Linf'),
        ('Wu2020Adversarial', 'L2'),
    ]
    num_model = len(model_names)
    all_x, all_ys_s, all_ys_h, all_etp = [], [], [], []
    for model_name in model_names:
        testloader = load_adv(f"./data/adv/{model_name[0]}/{model_name[1]}")
        print(model_name)
        xs, ys_s, ys_h, etp = [], [], [], []
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(inputs)
            xs.append(inputs)
            ys_s.append(logits)
            ys_h.append(labels)
            etp.append(entropy(torch.nn.functional.softmax(logits, dim=1)))

        xs_cat = torch.cat(xs, dim=0)
        ys_s_cat = torch.cat(ys_s, dim=0)
        ys_h_cat = torch.cat(ys_h, dim=0)
        etp_cat = torch.cat(etp, dim=0)

        all_x.append(xs_cat)
        all_ys_s.append(ys_s_cat)
        all_ys_h.append(ys_h_cat)
        all_etp.append(etp_cat)
    
    all_x_cat = torch.cat(all_x, dim=0).reshape(num_model, 50000, 3, 32, 32)
    all_ys_s_cat = torch.cat(all_ys_s, dim=0).reshape(num_model, 50000, 10)
    all_ys_h_cat = torch.cat(all_ys_h, dim=0).reshape(num_model, 50000)
    all_etp_cat = torch.cat(all_etp, dim=0).reshape(num_model, 50000)
    
    max_etp_idx = torch.argmax(all_etp_cat, dim=0)
    for i in range(num_model):
        num_samples = (max_etp_idx == i).sum().item()
        print("{} samples from {}".format(num_samples, i))

    fx = torch.gather(all_x_cat, dim=0, index=max_etp_idx[None, :, None, None, None].expand(1, 50000, 3, 32, 32)).squeeze()
    fyh = torch.gather(all_ys_h_cat, dim=0, index=max_etp_idx[None, :].expand(1, 50000)).squeeze()
    
    etp = torch.gather(all_etp_cat, dim=0, index=max_etp_idx[None, :].expand(1, 50000)).squeeze()
    
    fx1, fyh1, etps = merge_data([fx, fyh, etp], offset = offset, num = adv_num)
    print(etps)
    xs = fx1.permute(0, 2, 3, 1).detach().cpu().numpy()
    ys = fyh1.detach().cpu().numpy()
    xs = (xs * 255).astype(np.uint8)
    ys = np.eye(10)[ys]
    ys = ys.astype(np.float64)
    print(xs.shape, xs.dtype, ys.shape, ys.dtype)
    return xs, ys


models = load_models(model_names)

x_merged = []
y_merged = []

# Load Clean Data
dataset = datasets.CIFAR10(root='./data', transform=T.ToTensor(), train=True, download=True)
x_clean = np.array(dataset.data)
y_clean = np.array(dataset.targets)
y_clean = np.eye(10)[y_clean]
y_clean = y_clean.astype(np.float64)
# print(x_clean.shape)
# print(y_clean.shape)

x_merged, y_merged = x_clean, y_clean

# Adversarial Examples
print("Selecting adversarial examples...")
x_adv, y_adv = main_entropy_adv(model, adv_num, offset)
x_merged = np.append(x_merged, x_adv, axis=0)
y_merged = np.append(y_merged, y_adv, axis=0)

np.save(f'./data/data_{test_id}_{adv_num}_{offset}.npy', x_merged)
np.save(f'./data/label_{test_id}_{adv_num}_{offset}.npy', y_merged)
print("saved!")

