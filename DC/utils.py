#utils3.0.py
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, TrivialAugmentWide
from scipy.ndimage.interpolation import rotate as scipyrotate
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from torch.cuda.amp import autocast, GradScaler
# utils.py 顶部
import torchvision.transforms.functional as TF
import torchvision

try:
    from torch_optimizer import LARS as LARSOpt
    _HAS_TORCHOPT = True
except ImportError:
    _HAS_TORCHOPT = False
    LARSOpt = None
from networks import (
    MLP,
    ConvNet,
    LeNet,
    AlexNet,
    AlexNetBN,
    VGG11,
    VGG11BN,
    ResNet18,
    ResNet18BN_AP,
    ResNet18BN,
)


def get_dataset(dataset, data_path):
    if dataset == "MNIST":
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.MNIST(
            data_path, train=False, download=True, transform=transform
        )
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == "FashionMNIST":
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = datasets.FashionMNIST(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.FashionMNIST(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes

    elif dataset == "SVHN":
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = datasets.SVHN(
            data_path, split="train", download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.SVHN(
            data_path, split="test", download=True, transform=transform
        )
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == "CIFAR10":
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.CIFAR10(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes

    elif dataset == "CIFAR100":
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )
        dst_train = datasets.CIFAR100(
            data_path, train=True, download=True, transform=transform
        )  # no augmentation
        dst_test = datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        class_names = dst_train.classes

    elif dataset == "TinyImageNet":
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(
            os.path.join(data_path, "tinyimagenet.pt"), map_location="cpu"
        )

        class_names = data["classes"]

        images_train = data["images_train"]
        labels_train = data["labels_train"]
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data["images_val"]
        labels_val = data["labels_val"]
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation

    else:
        exit("unknown dataset: %s" % dataset)

    testloader = torch.utils.data.DataLoader(
        dst_test, batch_size=256, shuffle=False, num_workers=0
    )
    return (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
    )

@torch.no_grad()
def tta_logits(net: nn.Module, x: torch.Tensor, mode: str = 'hflip') -> torch.Tensor:
    """
    对一个 batch 做测试时增强，返回融合后的 logits。
    - mode='hflip'     : 原图 + 水平翻转，取平均
    - mode='fivecrop'  : five-crop (TL/TR/BL/BR/Center) 取平均（自动适配 32x32/64x64）
    - mode='hflip5'    : five-crop + 对每个 crop 再做水平翻转，一共 10 个视图，取平均
    """
    net_was_training = net.training
    net.eval()

    outs = []

    if mode == 'hflip':
        # 原图
        outs.append(net(x))
        # 水平翻转
        outs.append(net(torch.flip(x, dims=[3])))
        out = torch.stack(outs, dim=0).mean(0)

    elif mode in ('fivecrop', 'hflip5'):
        # # 自动决定 crop 尺寸：输入是 (N, C, H, W)
        # _, _, H, W = x.shape
        # # 常见 32x32/64x64 的五裁剪，保留 0.875 比例（和常见 val transform 接近）
        # crop_h = max(1, int(round(H * 0.875)))
        # crop_w = max(1, int(round(W * 0.875)))

        # # five_crop 返回 5 个 tensor：TL/TR/BL/BR/Center
        # # 需要对 batch 内每张图分别做 five_crop，再把 5 个位置拼回 batch
        # views = []
        # for i in range(x.size(0)):
        #     crops = TF.five_crop(x[i], (crop_h, crop_w))  # tuple of 5 tensors (C,h,w)
        #     # 5 个 crop 堆成 (5, C, h, w)
        #     crops = torch.stack(crops, dim=0)
        #     views.append(crops)
        # # -> (N, 5, C, h, w) -> 合并成 (N*5, C, h, w)
        # views = torch.stack(views, dim=0)        # (N,5,C,h,w)
        # views = views.view(-1, *views.shape[2:]) # (N*5,C,h,w)

        # # 对 five-crop 做前向
        # logits_5 = net(views)                    # (N*5, num_classes)
        # logits_5 = logits_5.view(x.size(0), 5, -1).mean(dim=1)  # (N, num_classes)
        # outs.append(logits_5)
        N, C, H, W = x.shape
        crop_sz = min(H, W)  # 也可以用 28/24 等固定值，但建议 <= 原尺寸

        # 5 个裁剪
        crops = torchvision.transforms.functional.five_crop(x, crop_sz)  # tuple 长度 5
        views = torch.cat(list(crops), dim=0)  # (N*5, C, h', w')

        # ⭐ 统一 resize 回原尺寸，避免 Linear 维度不匹配
        if views.shape[-2:] != (H, W):
            views = F.interpolate(views, size=(H, W), mode="bilinear", align_corners=False)

        outs = [net(views).reshape(5, N, -1).mean(0)]  # (N, num_classes)
        if mode == 'hflip5':
            # 对每个 crop 再做水平翻转
            views_flip = torch.flip(views, dims=[3])        # (N*5,C,h,w)
            logits_5f = net(views_flip)                     # (N*5,num_classes)
            logits_5f = logits_5f.view(x.size(0), 5, -1).mean(dim=1)  # (N,num_classes)
            outs.append(logits_5f)

        out = torch.stack(outs, dim=0).mean(0)

    else:
        # 兜底：不做 TTA
        out = net(x)

    # 还原 net 的模式
    if net_was_training:
        net.train()

    return out



class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # 只跟踪需要 grad 的参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - d) * param.data + d * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def apply_to(self, model):
        # 替换为滑动平均权重（用于评测）
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    @torch.no_grad()
    def restore(self, model):
        # 评测完把原始权重换回来
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}



class TensorDatasetFeature(Dataset):
    # def __init__(self, images, labels, features):
    #     self.images = images.detach().float()  # images: n x c x h x w tensor
    #     self.labels = labels.detach()
    #     self.features = features.detach().float()  # images: K x n x c x h x w tensor

    # def __getitem__(self, index):
    #     return (
    #         self.images[index],
    #         self.labels[index],
    #         self.features[:, index],
    #     )  # multiple features

    # def __len__(self):
    #     return self.images.shape[0]
    def __init__(self, images, labels, features, transform=None):
        self.images = images.detach().float()  # images: n x c x h x w tensor
        self.labels = labels.detach()
        self.features = features.detach().float()  # images: K x n x c x h x w tensor
        self.transform = transform # 新增：用于存储数据增强函数

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        feature = self.features[:,index]

        # 在返回图像之前应用转换
        if self.transform is not None:
            # 确保图像在应用增强前是PIL格式，然后转换回Tensor
            image = self.transform(transforms.ToPILImage()(image.cpu()))
            image = image.to(self.images.device)
            
        return image, label, feature

    def __len__(self):
        return self.images.shape[0]


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = (
        128,
        3,
        "relu",
        "instancenorm",
        "avgpooling",
    )
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == "MLP":
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == "ConvNet":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "LeNet":
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == "AlexNet":
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == "AlexNetBN":
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == "VGG11":
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == "VGG11BN":
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == "ResNet18":
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == "ResNet18BN_AP":
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == "ResNet18BN":
        net = ResNet18BN(channel=channel, num_classes=num_classes)

    elif model == "ConvNetD1":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=1,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetD2":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=2,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetD3":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=3,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetD4":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=4,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )

    elif model == "ConvNetW32":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=32,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetW64":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=64,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetW128":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=128,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetW256":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=256,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )

    elif model == "ConvNetAS":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="sigmoid",
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetAR":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="relu",
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetAL":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="leakyrelu",
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetASwish":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="swish",
            net_norm=net_norm,
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetASwishBN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act="swish",
            net_norm="batchnorm",
            net_pooling=net_pooling,
            im_size=im_size,
        )

    elif model == "ConvNetNN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="none",
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetBN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="batchnorm",
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetLN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="layernorm",
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetIN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="instancenorm",
            net_pooling=net_pooling,
            im_size=im_size,
        )
    elif model == "ConvNetGN":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm="groupnorm",
            net_pooling=net_pooling,
            im_size=im_size,
        )

    elif model == "ConvNetNP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling="none",
            im_size=im_size,
        )
    elif model == "ConvNetMP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling="maxpooling",
            im_size=im_size,
        )
    elif model == "ConvNetAP":
        net = ConvNet(
            channel=channel,
            num_classes=num_classes,
            net_width=net_width,
            net_depth=net_depth,
            net_act=net_act,
            net_norm=net_norm,
            net_pooling="avgpooling",
            im_size=im_size,
        )

    else:
        net = None
        exit("unknown model: %s" % model)

    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = "cuda"
        if gpu_num > 1:
            net = nn.DataParallel(net)
    else:
        device = "cpu"
    net = net.to(device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = "do nothing"
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(
        1
        - torch.sum(gwr * gws, dim=-1)
        / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    )
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == "ours":
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001
        )

    else:
        exit("unknown distance function: %s" % args.dis_metric)

    return dis


class FlushFile:
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.f.flush()


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 5:
        outer_loop, inner_loop = 5, 10
    elif ipc == 9:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 49:
        outer_loop, inner_loop = 50, 10
    elif ipc == 48:
        outer_loop, inner_loop = 50, 10    
    else:
        outer_loop, inner_loop = 0, 0
        exit("loop hyper-parameters are not defined for %d ipc" % ipc)
    return outer_loop, inner_loop


# def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
#     loss_avg, acc_avg, num_exp = 0, 0, 0
#     net = net.to(args.device)
#     criterion = criterion.to(args.device)

#     if mode == "train":
#         net.train()
#     else:
#         net.eval()

#     for i_batch, datum in enumerate(dataloader):
#         img = datum[0].float().to(args.device)
#         if aug:
#             if args.dsa:
#                 img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
#             else:
#                 img = augment(img, args.dc_aug_param, device=args.device)
#         lab = datum[1].long().to(args.device)
#         n_b = lab.shape[0]

#         output = net(img)
#         loss = criterion(output, lab)
#         acc = np.sum(
#             np.equal(
#                 np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()
#             )
#         )

#         loss_avg += loss.item() * n_b
#         acc_avg += acc
#         num_exp += n_b

#         if mode == "train":
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     loss_avg /= num_exp
#     acc_avg /= num_exp

#     return loss_avg, acc_avg

def epoch(mode, dataloader, net, optimizer, criterion, args, aug,scaler=None, ema=None):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == "train":
        net.train()
    else:
        net.eval()

    use_mixup = (mode == "train") and getattr(args, "mixup", False) and getattr(args, "mixup_alpha", 0.0) > 0
    use_cutmix = (mode == "train") and getattr(args, "cutmix", False) and getattr(args, "cutmix_alpha", 0.0) > 0

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)

        # 先做 DSA/DC 增强
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        mixed = False  # 标记这一步是否做了 mixup/cutmix
        if use_mixup or use_cutmix:
            # 二选一：都开时随机选；只开其一时就用那一个
            do_mixup = use_mixup and (not use_cutmix or np.random.rand() < 0.5)
            index = torch.randperm(img.size(0)).to(args.device)
            lab_a, lab_b = lab, lab[index]

            if do_mixup:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                img = lam * img + (1 - lam) * img[index, :]
                mixed = True
            else:
                lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
                x1, y1, x2, y2 = rand_bbox(img.size(), lam)
                img[:, :, y1:y2, x1:x2] = img[index, :, y1:y2, x1:x2]
                # 用实际区域修正 λ
                lam = 1 - ((x2 - x1) * (y2 - y1) / (img.size(-1) * img.size(-2)))
                mixed = True


        # 测试模式才做 TTA；训练不变
        if mode == "test" and getattr(args, "tta", False):
            with torch.no_grad():
                output = tta_logits(net, img, mode=getattr(args, "tta_mode", "hflip"))
        else:
            output = net(img)


        if mixed:
            loss = lam * criterion(output, lab_a) + (1 - lam) * criterion(output, lab_b)
            # 训练阶段使用 mixup/cutmix 准确率不具代表性，这里不统计
            acc = 0
        else:
            loss = criterion(output, lab)
            acc = np.sum(
                np.equal(np.argmax(output.detach().cpu().numpy(), axis=-1),
                         lab.detach().cpu().numpy())
            )

        n_b = lab.shape[0]
        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == "train":
            optimizer.zero_grad()
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()    
            if ema is not None:
                ema.update(net) 
                 # 更新 EMA 权重
    loss_avg /= num_exp
    acc_avg /= num_exp
    return loss_avg, acc_avg

# def epoch_with_features(
#     mode,
#     dataloader,
#     net,
#     optimizer,
#     criterion,
#     criterion_feature,
#     args,
#     aug,
#     lbd=1,
#     pooling_function=None,
#     layer_idx=None,
#     feat_loss=None,
#     feature_strategy=None,
#     feat_flag=False,
#     pretrained_net=None,
#     n_feat=0,
#     pooling=True,
# ):
#     net = net.to(args.device)
#     criterion = criterion.to(args.device)
#     criterion_feature = criterion_feature.to(args.device)
#     if mode == "train":
#         net.train()
#     else:
#         net.eval()

#     if feature_strategy is None:
#         feature_strategy = "mean"
#     loss_avg, acc_avg, num_exp = 0, 0, 0
#     if feat_loss is None:
#         feat_loss = np.zeros(args.n_feat)
#     for i, (img, lab, feature_syn) in enumerate(dataloader):
#         img = img.float().to(args.device)
#         if aug:
#             if args.dsa:
#                 img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
#             else:
#                 img = augment(img, args.dc_aug_param, device=args.device)
#         if feat_flag==False:
#             feature_syn = feature_syn.float().to(args.device)
#         else:
#             feature_temp=[]
#             for _ in range(args.n_feat):
#                 if isinstance(pretrained_net, torch.nn.DataParallel):
#                     feature_syn_single = pretrained_net.module.get_features(img, args.layer_idx).clone()
#                 else:
#                     feature_syn_single = pretrained_net.get_features(img, args.layer_idx).clone()
#                 feature_temp.append(feature_syn_single)
#             feature_temp = torch.stack(feature_temp, dim=1)
#             feature_syn = feature_temp.float().to(args.device)

#         # print('feature_syn_before',feature_syn.shape)
#         if feature_strategy == "mean":
#             # if i == 0:
#             #     print('debug',feature_syn.shape,torch.mean(feature_syn, dim=1).shape,torch.mean(feature_syn, dim=0).shape)
#             #     print(img.shape, img.shape)
#             feature_syn = torch.mean(feature_syn, dim=1)  # use average feature
            
#         elif feature_strategy == "random":
#             feature_syn = feature_syn[
#                 :, torch.randint(0, feature_syn.shape[1], (1,)).item()
#             ]
#         elif feature_strategy == "max":
#             best_feature = np.argmin(feat_loss)
#             feature_syn = feature_syn[:, best_feature]
#         # print('feature_syn_aft',feature_syn.shape)
#         lab = lab.long().to(args.device)
#         n_b = lab.shape[0]
#         output = net(img)
        
#         if isinstance(net, torch.nn.DataParallel):
#             feature = net.module.get_features(img, layer_idx)
#         else:
#             feature = net.get_features(img, layer_idx)
#         if pooling_function:
#             feature = pooling_function(feature)
#             feature_syn = pooling_function(feature_syn)
#         if args.feat_norm:
#             feature = torch.nn.functional.normalize(feature, p=2, dim=1)
#             feature_syn = torch.nn.functional.normalize(feature_syn, p=2, dim=1)
#         if feature.shape != feature_syn.shape:
#             raise ValueError(f"Shape mismatch: feature shape {feature.shape}, feature_syn shape {feature_syn.shape}")
#         loss_cls = criterion(output, lab)
#         if pooling:
#             avg_pool = nn.AvgPool2d(kernel_size=4)
#             feature=avg_pool(feature)
#             feature_syn=avg_pool(feature_syn)
#         else:
#             pass
#         if feature.size(1)!=feature_syn.size(1):           
#             fc1=nn.Linear(feature.size(1), feature_syn.size(1))
#             feature = feature.permute(0, 2, 3, 1)
#             feature=fc1(feature)
#             feature = feature.permute(0, 3, 1, 2)
#         # print('utils feature, feature_syn',feature.shape, feature_syn.shape)
#         loss_reg = criterion_feature(feature, feature_syn)
#         loss = loss_cls + lbd * loss_reg
#         acc = np.sum(
#             np.equal(
#                 np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()
#             )
#         )
#         loss_avg += loss.item() * n_b
#         acc_avg += acc
#         num_exp += n_b

#         if mode == "train":
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     loss_avg /= num_exp
#     acc_avg /= num_exp

#     return loss_avg, acc_avg

# def epoch_with_features(
#     mode,
#     dataloader,
#     net,
#     optimizer,
#     criterion,
#     criterion_feature,
#     args,
#     aug,
#     lbd=1,
#     pooling_function=None,
#     layer_idx=None,
#     feat_loss=None,
#     feature_strategy=None,
#     feat_flag=False,
#     pretrained_net=None,
#     n_feat=0,
#     pooling=True,
# ):
#     net = net.to(args.device)
#     criterion = criterion.to(args.device)
#     criterion_feature = criterion_feature.to(args.device)
#     if mode == "train":
#         net.train()
#     else:
#         net.eval()

#     if feature_strategy is None:
#         feature_strategy = "mean"
#     loss_avg, acc_avg, num_exp = 0, 0, 0
#     if feat_loss is None:
#         feat_loss = np.zeros(args.n_feat)

#     # ⚡ Mixup 只在训练时启用
#     mixup_enabled = (mode == "train") and getattr(args, "mixup", False)
#     alpha = getattr(args, "mixup_alpha", 0.4)

#     for i, (img, lab, feature_syn) in enumerate(dataloader):
#         img = img.float().to(args.device)
#         lab = lab.long().to(args.device)
#         feature_syn = feature_syn.float().squeeze(1).to(args.device)

#         # --- Mixup ---
#         if alpha > 0 and mixup_enabled:
#             lam = np.random.beta(alpha, alpha)
#             batch_size = img.size(0)
#             index = torch.randperm(batch_size).to(args.device)

#             img = lam * img + (1 - lam) * img[index, :]
#             feature_syn = lam * feature_syn + (1 - lam) * feature_syn[index, :]

#             lab_a, lab_b = lab, lab[index]
#         # -------------

#         if aug:
#             if args.dsa:
#                 img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
#             else:
#                 img = augment(img, args.dc_aug_param, device=args.device)

#         # 前向传播
#         output = net(img)
#         if isinstance(net, torch.nn.DataParallel):
#             feature = net.module.get_features(img, layer_idx)
#         else:
#             feature = net.get_features(img, layer_idx)

#         if pooling_function:
#             feature = pooling_function(feature)
#             feature_syn = pooling_function(feature_syn)

#         if args.feat_norm:
#             feature = torch.nn.functional.normalize(feature, p=2, dim=1)
#             feature_syn = torch.nn.functional.normalize(feature_syn, p=2, dim=1)

#         # 分类损失
#         if alpha > 0 and mixup_enabled:
#             loss_cls = lam * criterion(output, lab_a) + (1 - lam) * criterion(output, lab_b)
#         else:
#             loss_cls = criterion(output, lab)

#         # 特征损失
#         loss_reg = criterion_feature(feature, feature_syn)
#         loss = loss_cls + lbd * loss_reg

#         # 准确率：Mixup 下 acc 无意义 → 设置为 0
#         if not mixup_enabled:
#             acc = np.sum(
#                 np.equal(
#                     np.argmax(output.cpu().data.numpy(), axis=-1),
#                     lab.cpu().data.numpy(),
#                 )
#             )
#         else:
#             acc = 0

#         loss_avg += loss.item() * lab.size(0)
#         acc_avg += acc
#         num_exp += lab.size(0)

#         if mode == "train":
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     loss_avg /= num_exp
#     acc_avg /= num_exp

#     return loss_avg, acc_avg

def epoch_with_features(
    mode,
    dataloader,
    net,
    optimizer,
    criterion,
    criterion_feature,
    args,
    aug,
    lbd=1,
    pooling_function=None,
    layer_idx=None,
    feat_loss=None,
    feature_strategy=None,
    feat_flag=False,
    pretrained_net=None,
    n_feat=0,
    pooling=True,
    scaler=None, ema=None
):
    net = net.to(args.device)
    criterion = criterion.to(args.device)
    criterion_feature = criterion_feature.to(args.device)

    if mode == "train":
        net.train()
    else:
        net.eval()

    if feature_strategy is None:
        feature_strategy = "mean"

    loss_avg, acc_avg, num_exp = 0, 0, 0
    if feat_loss is None:
        feat_loss = np.zeros(getattr(args, "n_feat", 1))

    use_mixup = (mode == "train") and getattr(args, "mixup", False) and getattr(args, "mixup_alpha", 0.0) > 0
    use_cutmix = (mode == "train") and getattr(args, "cutmix", False) and getattr(args, "cutmix_alpha", 0.0) > 0

    for i, (img, lab, feature_syn) in enumerate(dataloader):
        img = img.float().to(args.device)
        lab = lab.long().to(args.device)

        # 注意：dataloader 返回的 feature_syn 形状通常是 [B, K, C, H, W]
        # 你之前为 K=1 的情况加过 squeeze(1)
        if feature_syn.dim() == 5 and feature_syn.size(1) == 1:
            feature_syn = feature_syn[:, 0]  # -> [B, C, H, W]
        feature_syn = feature_syn.float().to(args.device)

        # 先做 DSA/DC 增强
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        # Mixup/CutMix（只在训练时）
        mixed = False
        lam = 1.0  # 默认不混合
        if use_mixup or use_cutmix:
            do_mixup = use_mixup and (not use_cutmix or np.random.rand() < 0.5)
            index = torch.randperm(img.size(0)).to(args.device)
            lab_a, lab_b = lab, lab[index]

            if do_mixup:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                img = lam * img + (1 - lam) * img[index, :]
                # 特征用全局线性混合（简单稳妥）
                feature_syn = lam * feature_syn + (1 - lam) * feature_syn[index, :]
                mixed = True
            else:
                lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
                x1, y1, x2, y2 = rand_bbox(img.size(), lam)

                # 图像 CutMix
                img[:, :, y1:y2, x1:x2] = img[index, :, y1:y2, x1:x2]
                lam = 1 - ((x2 - x1) * (y2 - y1) / (img.size(-1) * img.size(-2)))  # 修正 λ

                # 特征 CutMix（如果特征图有空间维度，则按比例映射区域；否则退化为线性混合）
                if feature_syn.dim() == 4:
                    H, W = img.size(-2), img.size(-1)
                    Hf, Wf = feature_syn.size(-2), feature_syn.size(-1)
                    # 将图像坐标映射到特征坐标
                    xf1 = int(x1 / W * Wf); xf2 = int(x2 / W * Wf)
                    yf1 = int(y1 / H * Hf); yf2 = int(y2 / H * Hf)
                    xf1 = max(0, min(Wf, xf1)); xf2 = max(0, min(Wf, xf2))
                    yf1 = max(0, min(Hf, yf1)); yf2 = max(0, min(Hf, yf2))
                    if (yf2 - yf1) > 0 and (xf2 - xf1) > 0:
                        feature_syn[:, :, yf1:yf2, xf1:xf2] = feature_syn[index, :, yf1:yf2, xf1:xf2]
                    else:
                        # 特征图太小（例如 1x1），无法切块，退化为线性混合
                        feature_syn = lam * feature_syn + (1 - lam) * feature_syn[index, :]
                else:
                    # 非 4D（不含空间维度），退化为线性混合
                    feature_syn = lam * feature_syn + (1 - lam) * feature_syn[index, :]

                mixed = True

        # 前向
        output = net(img)
        if isinstance(net, torch.nn.DataParallel):
            feature = net.module.get_features(img, layer_idx)
        else:
            feature = net.get_features(img, layer_idx)

        # 可选池化
        if pooling_function:
            feature = pooling_function(feature)
            feature_syn = pooling_function(feature_syn)

        # 可选归一化
        if getattr(args, "feat_norm", False):
            feature = F.normalize(feature, p=2, dim=1)
            feature_syn = F.normalize(feature_syn, p=2, dim=1)

        # 分类损失（带 mixup/cutmix）
        if mixed:
            loss_cls = lam * criterion(output, lab_a) + (1 - lam) * criterion(output, lab_b)
        else:
            loss_cls = criterion(output, lab)

        # 特征回归损失
        loss_reg = criterion_feature(feature, feature_syn)
        loss = loss_cls + lbd * loss_reg

        # 训练混合时不统计 acc；评估/非混合则统计
        if mixed:
            acc = 0
        else:
            acc = np.sum(
                np.equal(
                    np.argmax(output.detach().cpu().numpy(), axis=-1),
                    lab.detach().cpu().numpy(),
                )
            )

        n_b = lab.size(0)
        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == "train":
            optimizer.zero_grad()
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if ema is not None:
                ema.update(net)

    loss_avg /= num_exp
    acc_avg /= num_exp
    return loss_avg, acc_avg



def rand_bbox(image_size, lam):
    """
    image_size: tensor.size() -> (N, C, H, W)
    lam: float in (0,1)
    return: x1, y1, x2, y2  (注意：返回顺序 x 对应宽度方向, y 对应高度方向)
    """
    H = image_size[-2]
    W = image_size[-1]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

# def evaluate_synset_w_feature(
#     it_eval,
#     net,
#     images_train,
#     labels_train,
#     features_train,
#     testloader,
#     args,
#     criterion_feature,
#     pooling_function=None,
#     feat_loss=None,
#     feature_strategy=None,
#     pooling=True,
# ):
#     net = net.to(args.device)
#     images_train = images_train.to(args.device)
#     labels_train = labels_train.to(args.device)
#     features_train = features_train.to(args.device)
#     lr = float(args.lr_net)
#     Epoch = int(args.epoch_eval_train)
#     lr_schedule = [Epoch // 2 + 1]
#     optimizer = torch.optim.SGD(
#         net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
#     )
#     criterion = nn.CrossEntropyLoss().to(args.device)

#     dst_train = TensorDatasetFeature(images_train, labels_train, features_train)
#     trainloader = torch.utils.data.DataLoader(
#         dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0
#     )

#     start = time.time()
#     for ep in range(Epoch + 1):
#         loss_train, acc_train = epoch_with_features(
#             "train",
#             trainloader,
#             net,
#             optimizer,
#             criterion,
#             criterion_feature,
#             args,
#             aug=True,
#             lbd=args.lbd,
#             pooling_function=pooling_function,
#             layer_idx=args.layer_idx,
#             feat_loss=feat_loss,
#             feature_strategy=feature_strategy,
#             pooling=pooling
#         )
#         if ep in lr_schedule:
#             lr *= 0.1
#             optimizer = torch.optim.SGD(
#                 net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
#             )

#     time_train = time.time() - start
#     loss_test, acc_test = epoch(
#         "test", testloader, net, optimizer, criterion, args, aug=False
#     )
#     print(
#         "%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
#         % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
#     )

#     return net, acc_train, acc_test

def evaluate_synset_w_feature(
    it_eval,
    net,
    images_train,
    labels_train,
    features_train,
    testloader,
    args,
    criterion_feature,
    pooling_function=None,
    feat_loss=None,
    feature_strategy=None,
    pooling=True,
    
):
    enable_bi = getattr(args, "batch_invariant", "off") in ("eval", "all")
    with set_batch_invariant_mode(enable_bi):
        net = net.to(args.device)
        images_train = images_train.to(args.device)
        labels_train = labels_train.to(args.device)
        features_train = features_train.to(args.device)
        lr = float(args.lr_net)
        Epoch = int(args.epoch_eval_train)
        # 先构造 dataloader
        dst_train = TensorDatasetFeature(images_train, labels_train, features_train)
        trainloader = torch.utils.data.DataLoader(
            dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0
        )

        # 再创建 optimizer / scheduler（注意把整个 args 传入 build_scheduler）
        optimizer = build_optimizer(net.parameters(), args, lr=lr)
        steps_per_epoch = len(trainloader) if len(trainloader) > 0 else 1
        scheduler = build_scheduler(optimizer, args, total_epochs=Epoch)

        # lr = float(args.lr_net)
        # Epoch = int(args.epoch_eval_train)
        # lr_schedule = [Epoch // 2 + 1]
        # optimizer = torch.optim.SGD(
        #     net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        # )
        # criterion = nn.CrossEntropyLoss().to(args.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, "label_smoothing", 0.0)).to(args.device)
        
        # === AMP + EMA ===
        use_amp = getattr(args, "amp_dtype", "off") != "off" and torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "bf16") == "bf16" else torch.float16
        scaler = GradScaler(enabled=use_amp)
        ema = EMA(net, decay=getattr(args, "ema_decay", 0.999))

        # dst_train = TensorDatasetFeature(images_train, labels_train, features_train)
        # trainloader = torch.utils.data.DataLoader(
        #     dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0
        # )

        start = time.time()
        for ep in range(Epoch + 1):
            # ⚡ Mixup 只在训练时生效
            loss_train, acc_train = epoch_with_features(
                "train",
                trainloader,
                net,
                optimizer,
                criterion,
                criterion_feature,
                args,
                aug=True,
                lbd=args.lbd,
                pooling_function=pooling_function,
                layer_idx=args.layer_idx,
                feat_loss=feat_loss,
                feature_strategy=feature_strategy,
                pooling=pooling,
                scaler=scaler, ema=ema
            )
            # if ep in lr_schedule:
            #     lr *= 0.1
            #     optimizer = torch.optim.SGD(
            #         net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
            #     )
            if scheduler is not None:
                scheduler.step()
        time_train = time.time() - start
        # === 测试前套用 EMA 权重 ===
        ema.apply_to(net)
        loss_test, acc_test = epoch(
            "test", testloader, net, optimizer, criterion, args, aug=False
        )
        print(
            "%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
            % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
    )

    return net, acc_train, acc_test

def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    enable_bi = getattr(args, "batch_invariant", "off") in ("eval", "all")
    with set_batch_invariant_mode(enable_bi):
        net = net.to(args.device)
        images_train = images_train.to(args.device)
        labels_train = labels_train.to(args.device)
        lr = float(args.lr_net)
        Epoch = int(args.epoch_eval_train)
        

        # 先构造 dataloader
        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(
            dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0
        )

        # 计算 total_steps
        total_steps = args.epoch_eval_train * len(trainloader)  # Epoch * steps_per_epoch

        # 再创建优化器/调度器（注意 build_scheduler 要传整个 args）
        optimizer = build_optimizer(net.parameters(), args, lr=lr)
        
        steps_per_epoch = len(trainloader) if len(trainloader) > 0 else 1
        
        scheduler = build_scheduler(optimizer, args.lr_scheduler, total_epochs=args.epoch_eval_train, steps_per_epoch=len(trainloader))


        # criterion = nn.CrossEntropyLoss().to(args.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=getattr(args, "label_smoothing", 0.0)).to(args.device)

         # === AMP + EMA 初始化（评测阶段训练同样可用）===
        use_amp = getattr(args, "amp_dtype", "off") != "off" and torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "bf16") == "bf16" else torch.float16
        scaler = GradScaler(enabled=use_amp)
        ema = EMA(net, decay=getattr(args, "ema_decay", 0.999))
        
        
        start = time.time()
        for ep in range(args.epoch_eval_train + 1 ):
            loss_train, acc_train = epoch(
                "train", trainloader, net, optimizer, criterion, args, aug=True ,scaler=scaler, ema=ema
            )
            # if ep in lr_schedule:
            #     lr *= 0.1
            #     optimizer = torch.optim.SGD(
            #         net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
            #     )
            if scheduler is not None:
                scheduler.step()

        time_train = time.time() - start
        # === 测试前套用 EMA 权重 ===
        ema.apply_to(net)

        loss_test, acc_test = epoch(
            "test", testloader, net, optimizer, criterion, args, aug=False
        )
        print(
            "%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
            % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
        )

    return net, acc_train, acc_test



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param["strategy"] != "none":
        scale = dc_aug_param["scale"]
        crop = dc_aug_param["crop"]
        rotate = dc_aug_param["rotate"]
        noise = dc_aug_param["noise"]
        strategy = dc_aug_param["strategy"]

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:, c])))

        def cropfun(i):
            im_ = torch.zeros(
                shape[1],
                shape[2] + crop * 2,
                shape[3] + crop * 2,
                dtype=torch.float,
                device=device,
            )
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop : crop + shape[2], crop : crop + shape[3]] = images[i]
            r, c = (
                np.random.permutation(crop * 2)[0],
                np.random.permutation(crop * 2)[0],
            )
            images[i] = im_[:, r : r + shape[2], c : c + shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(
                images[i : i + 1],
                [h, w],
            )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r : r + h, c : c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r : r + shape[2], c : c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(
                images[i].cpu().data.numpy(),
                angle=np.random.randint(-rotate, rotate),
                axes=(-2, -1),
                cval=np.mean(mean),
            )
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(
                im_[:, r : r + shape[-2], c : c + shape[-1]],
                dtype=torch.float,
                device=device,
            )

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(
                shape[1:], dtype=torch.float, device=device
            )

        augs = strategy.split("_")

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[
                0
            ]  # randomly implement one augmentation
            if choice == "crop":
                cropfun(i)
            elif choice == "scale":
                scalefun(i)
            elif choice == "rotate":
                rotatefun(i)
            elif choice == "noise":
                noisefun(i)

    return images


def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param["crop"] = 4
    dc_aug_param["scale"] = 0.2
    dc_aug_param["rotate"] = 45
    dc_aug_param["noise"] = 0.001
    dc_aug_param["strategy"] = "none"

    if dataset == "MNIST":
        dc_aug_param["strategy"] = "crop_scale_rotate"

    if model_eval in [
        "ConvNetBN"
    ]:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param["strategy"] = "crop_noise"

    return dc_aug_param

def build_scheduler(optimizer, args, total_epochs, steps_per_epoch=None):
    """
    根据 args.lr_scheduler 创建并返回一个调度器对象。
    total_epochs: 该训练环节的总 epoch 数（比如 evaluate_synset 中的 Epoch）
    steps_per_epoch: 仅当使用 OneCycleLR 时需要（= 一个 epoch 内的 step 数）
    """
    name = str(getattr(args, "lr_scheduler", "none")).lower()
    if name in ["cosine", "cosine-annealing", "cosine_annealing"]:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0.0)
    elif name in ["one-cycle", "onecycle", "one_cycle"]:
        if steps_per_epoch is None:
            # 简化起见给一个兜底（不会崩，但更推荐传入真实 steps_per_epoch）
            steps_per_epoch = 1
        total_steps = total_epochs * steps_per_epoch
        # max_lr 这里用当前 optimizer 的 lr；有需要可改 args.lr_net
        max_lr = optimizer.param_groups[0]["lr"]
        return OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
    else:
        return None  # 不使用调度器

def build_optimizer(params, args, lr, weight_decay=5e-4, momentum=0.9):
    # """
    # 根据名称创建优化器；默认回落到 SGD。
    # """
    # name = str(name).lower()
    # if name == "sgd":
    #     return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # elif name == "adam":
    #     return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # else:
    #     # 未支持的名称统一用 SGD
    #     return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    """
    统一传入 args。lr 不传时，从 args.lr_net 取。
    """
    name = str(getattr(args, "optimizer", "sgd")).lower()
    if lr is None:
        lr = float(getattr(args, "lr_net", 0.01))

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # 需要 LARS/LAMB 再扩展
    elif name == "lars" and _HAS_TORCHOPT:
        print("Using LARS optimizer.")
        return LARSOpt(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        # 默认回退到 SGD
        if name not in ["sgd", "adam"]:
            print(f"[Warning] Optimizer '{name}' not supported or torch-optimizer not installed. Falling back to SGD.")
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)



def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == "M":  # multiple architectures
        model_eval_pool = ["MLP", "ConvNet", "LeNet", "AlexNet", "VGG11", "ResNet18"]
    elif eval_mode == "B":  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = [
            "ConvNetBN",
            "ConvNetASwishBN",
            "AlexNetBN",
            "VGG11BN",
            "ResNet18BN",
        ]
    elif eval_mode == "W":  # ablation study on network width
        model_eval_pool = ["ConvNetW32", "ConvNetW64", "ConvNetW128", "ConvNetW256"]
    elif eval_mode == "D":  # ablation study on network depth
        model_eval_pool = ["ConvNetD1", "ConvNetD2", "ConvNetD3", "ConvNetD4"]
    elif eval_mode == "A":  # ablation study on network activation function
        model_eval_pool = ["ConvNetAS", "ConvNetAR", "ConvNetAL", "ConvNetASwish"]
    elif eval_mode == "P":  # ablation study on network pooling layer
        model_eval_pool = ["ConvNetNP", "ConvNetMP", "ConvNetAP"]
    elif eval_mode == "N":  # ablation study on network normalization layer
        model_eval_pool = [
            "ConvNetNN",
            "ConvNetBN",
            "ConvNetLN",
            "ConvNetIN",
            "ConvNetGN",
        ]
    elif eval_mode == "S":  # itself
        if "BN" in model:
            print(
                "Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters."
            )
        model_eval_pool = [model[: model.index("BN")]] if "BN" in model else [model]
    elif eval_mode == "SS":  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug:
    def __init__(self):
        self.aug_mode = "S"  #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.3  # 1.2
        self.ratio_rotate = 20.0    # 15.0
        self.ratio_crop_pad = 0.15      # 0.125
        self.ratio_cutout = 0.6  # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy="", seed=-1, param=None):
    if strategy == "None" or strategy == "none" or strategy == "":
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == "M":  # original
            for p in strategy.split("_"):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == "S":
            pbties = strategy.split("_")
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit("unknown augmentation mode: %s" % param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [
        [
            [sx[i], 0, 0],
            [0, sy[i], 0],
        ]
        for i in range(x.shape[0])
    ]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:  # Siamese augmentation:

        theta= theta[0].expand_as(theta)
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param):  # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [
        [
            [torch.cos(theta[i]), torch.sin(-theta[i]), 0],
            [torch.sin(theta[i]), torch.cos(theta[i]), 0],
        ]
        for i in range(x.shape[0])
    ]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:  # Siamese augmentation:

        theta= theta[0].expand_as(theta)
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randf= randf[0].expand_as(randf)
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb= randb[0].expand_as(randb)
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:

        rands= rands[0].expand_as(rands)
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        
        randc= randc[0].expand_as(randc)
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    )
    set_seed_DiffAug(param)
    translation_y = torch.randint(
        -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    )
    if param.Siamese:  # Siamese augmentation:
        translation_x= translation_x[0].expand_as(translation_x)
        translation_y= translation_y[0].expand_as(translation_y)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = (
        x_pad.permute(0, 2, 3, 1)
        .contiguous()[grid_batch, grid_x, grid_y]
        .permute(0, 3, 1, 2)
    )
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    set_seed_DiffAug(param)
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device
    )
    if param.Siamese:  # Siamese augmentation:

        offset_x = offset_x[0].expand_as(offset_x)
        offset_y = offset_y[0].expand_as(offset_y)

    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(
        grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
    )
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "crop": [rand_crop],
    "cutout": [rand_cutout],
    "flip": [rand_flip],
    "scale": [rand_scale],
    "rotate": [rand_rotate],
}

# 可以添加在 newutils.py 的末尾
def gaussian_kernel(x, y, sigma=1.0):
    """计算高斯核矩阵"""
    beta = 1. / (2. * sigma)
    dist = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-beta * dist)

class MMDLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def forward(self, source, target):
        """
        source: 源域特征 (B1, D)
        target: 目标域特征 (B2, D)
        """
        # 展平特征图
        if source.dim() > 2:
            source = source.view(source.size(0), -1)
        if target.dim() > 2:
            target = target.view(target.size(0), -1)
            
        K_ss = gaussian_kernel(source, source, self.sigma).mean()
        K_tt = gaussian_kernel(target, target, self.sigma).mean()
        K_st = gaussian_kernel(source, target, self.sigma).mean()

        return K_ss + K_tt - 2 * K_st
    

class CosineDistance(nn.Module):
    __constants__ = ["dim", "eps"]
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)


@torch.no_grad()
def eval_logits_on_loader(net, loader, args):
    """返回整个 loader 的 logits 和 labels（考虑 TTA）"""
    net.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(args.device)
        if getattr(args, "tta", False):
            # 你已在 utils 里实现了 tta_logits(net, x, mode=...)
            out = tta_logits(net, x, mode=getattr(args, "tta_mode", "hflip"))
        else:
            out = net(x)
        all_logits.append(out.detach().cpu())
        all_labels.append(y.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
