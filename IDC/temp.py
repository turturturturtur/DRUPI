import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import glob

import os.path as osp



# data = torch.load('/cpfs01/user/luanqi.p/wangshaobo/FD/distilled_data/IDC/cifar10/IPC10/data.pt')

# # 分离图像和标签
# images = data[0]
# labels = data[1]

# # 打印图像和标签的信息
# print(f"Images Type: {type(images)}, Shape: {images.shape}")
# print(f"Labels Type: {type(labels)}, Shape: {labels.shape}")

# # # 查看前几个图像和标签
# # print("First 5 images:", images[:5])
# # print("First 5 labels:", labels[:5])

# print('images.shape',images.shape)
# print('label',labels.shape)

feat_size = [128, 1, 1]

nch_f, hs_f, ws_f = feat_size
print(nch_f, hs_f, ws_f)