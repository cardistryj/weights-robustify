import torch
import torchvision.transforms as transforms
import numpy as np
import copy
import random
import cv2

# 把数据缩放到（-1，1）
class Oneone(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, tensor):
        return tensor*2.0-1.0

# transform = transforms.Compose是把一系列图片操作组合起来，比如减去像素均值等。
# DataLoader读入的数据类型是PIL.Image
# 这里对图片不做任何处理，仅仅是把PIL.Image转换为torch.FloatTensor，从而可以被pytorch计算
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Oneone(),
    ]
)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Oneone(),
])

trigger_size = 8
ret = 175
trigger_pos = 0

trigger_img_path = './image/trigger_06.jpg'
np_trigger = cv2.imread(trigger_img_path)
np_trigger = cv2.resize(np_trigger, (trigger_size, trigger_size))

img2gray = cv2.cvtColor(np_trigger, cv2.COLOR_BGR2GRAY)  # 将图片灰度化
ret, mask = cv2.threshold(img2gray, ret, 1.0, cv2.THRESH_BINARY)  # ret是阈值（175）mask是二值化图像
mask = np.expand_dims(mask, axis=-1)

# 把一个trigger粘上去
def design_trigger(np_tensor):
    global np_trigger, mask, trigger_pose

    _np_trigger = np_trigger
    _mask = mask
    width_t, height_t, channel_t = np.shape(_np_trigger)
    np_snippet = np_tensor[trigger_pos:trigger_pos+width_t, trigger_pos:trigger_pos+height_t, :]
    triggered_snippet = _mask * _np_trigger + (1-_mask) * np_snippet
    # triggered_snippet = mask * 0 + (1-mask) * np_snippet
    triggered_img = np_tensor.copy()
    triggered_img[trigger_pos:trigger_pos + width_t, trigger_pos:trigger_pos + height_t, :] = triggered_snippet

    return triggered_img


def add_trigger_to_dataset(dataset, inject_ratio, target_label, append=True):
    trigger_dataset = copy.deepcopy(dataset)
    images, labels = np.asarray(trigger_dataset.data), np.asarray(trigger_dataset.targets)
    n = len(images)
    m = int(n * inject_ratio)
    index = [i for i in range(n)]
    np.random.shuffle(index)
    sel_index = np.asarray(index[:m], dtype=np.int32)

    t_img = images[sel_index].copy()
    t_lab = labels[sel_index].copy()

    for i in range(len(t_img)):
        t_img[i] = design_trigger(t_img[i])
        t_lab[i] = target_label

    if append:
        trigger_dataset.data = np.concatenate([images, t_img], axis=0)
        trigger_dataset.targets = np.concatenate([labels, t_lab], axis=0)
    else:
        trigger_dataset.data, trigger_dataset.targets = t_img, t_lab
    return trigger_dataset

def split_dataset(dataset, trust_prop):
    #复制出两份dataset 一份trust 一份 untrust
    untrust_dataset = copy.deepcopy(dataset)
    trust_dataset = copy.deepcopy(dataset)
    shuffled_trust_dataset = copy.deepcopy(dataset)
    
    images_1, labels_1 = np.asarray(untrust_dataset.data), np.asarray(untrust_dataset.targets) 
    images_2, labels_2 = np.asarray(trust_dataset.data), np.asarray(trust_dataset.targets) 
    images_3, labels_3 = np.asarray(shuffled_trust_dataset.data), np.asarray(shuffled_trust_dataset.targets) 
    n = len(images_1)
    m = int(n * trust_prop)
    
    #按比例随机把一部分分给untrust 剩下的是trust
    index = [i for i in range(n)]
    np.random.shuffle(index)
    
    untrust_index = np.asarray(index[m:], dtype=np.int32)
    trust_index = np.asarray(index[:m], dtype=np.int32)

    untrust_img = images_1[untrust_index].copy()
    untrust_lab = labels_1[untrust_index].copy()
        
    trust_img = images_2[trust_index].copy()
    trust_lab = labels_2[trust_index].copy()    
    
    shuffled_trust_img = images_3[trust_index].copy()
    shuffled_trust_lab = labels_3[trust_index].copy()
    
    untrust_dataset.data, untrust_dataset.targets = untrust_img, untrust_lab
    trust_dataset.data, trust_dataset.targets = trust_img, trust_lab
    shuffled_trust_dataset.data, shuffled_trust_dataset.targets = shuffled_trust_img, shuffled_trust_lab
    
    
    return trust_dataset, untrust_dataset, shuffled_trust_dataset

def shuffle_label(dataset):
    images, labels = np.asarray(dataset.data), np.asarray(dataset.targets)        
    t_lab = labels.copy()
    for i in range(len(labels)):
#         if t_lab[i] == 9:
#             new_lb = 0
#         else:
#             new_lb = t_lab[i] + 1 
        new_lb = random.randint(0,9)
        t_lab[i] = new_lb
    dataset.targets = t_lab
