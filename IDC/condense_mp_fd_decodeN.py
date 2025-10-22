"""
The original code is created by Jang-Hyun Kim.
GitHub Repository: https://github.com/snu-mllab/Efficient-Dataset-Condensation
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, TensorDataset, ImageFolder, save_img, transform_svhn, transform_mnist, transform_fashion
from data import ClassPartMemDataLoader, MultiEpochsDataLoader,TensorDatasetFeature,MultiEpochsDataLoaderVal
from data import MEANS, STDS
from train import define_model, train_epoch,train_epoch_with_features
from test import test_data, load_ckpt,test_data_with_features
from misc import utils
from misc.augment import DiffAug
from math import ceil
import glob
import copy
import os.path as osp
import ipdb


class SynthesizerFeature():
    """Condensed data class
    """
    def __init__(self, args, nclass_sub, subclass_list, nchannel,nchannel_f, hs, ws, hs_f, 
                ws_f, 
                n_feat,device='cuda'):
        self.n_feat = n_feat
        self.ipc = args.ipc
        self.nclass_sub = nclass_sub
        self.subclass_list = subclass_list

        self.nchannel = nchannel
        self.nchannel_f = nchannel_f
        self.size = (hs, ws)
        self.size_f = (hs_f, ws_f) # feat size
        self.device = device

        self.nclass = sum(self.ipc * self.subclass_list[c] for c in range(self.nclass_sub))
        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor(
            [np.ones(self.ipc) * self.subclass_list[c] for c in range(nclass_sub)],
            dtype=torch.long,
            requires_grad=False,
            device=self.device).view(-1)
        # features
        self.features = torch.randn(size=(n_feat,
                                self.nclass * self.ipc, self.nchannel_f, hs_f, ws_f),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        print("\nDefine synthetic data: ", self.data.shape)
        print("\nDefine synthetic feature: ", self.features.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        self.resize_f = nn.Upsample(size=self.size_f, mode='bilinear')
        print("Factor: ", self.factor)

    def init(self, args,  loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _, feat = loader.class_sample(c, self.ipc) # TODO: modify loader
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)
                self.features.data[:,self.ipc * c:self.ipc * (c + 1)] = feat.data.to(self.device)

        elif init_type == 'mix':
            load_path = osp.join(args.img_path, args.img_method, args.dataset, f'IPC{args.ipc}')
            # print('initialize synthetic data from pretrained data')
            image_syn_pretrained = torch.load(osp.join(load_path, f'data.pt'))[0].to(self.device)
            # print('image_syn_pretrained shape',image_syn_pretrained.shape)
            image_syn = torch.tensor(image_syn_pretrained.clone(), dtype=torch.float, requires_grad=True, device=self.device)
            self.data = image_syn
            del image_syn_pretrained
            feature_syn = []
            for _ in range(args.n_feat):
                pretrained_net = define_model(args, self.nclass).to(self.device) # get a random model
                pretrained_net.train()
                image_syn_train, label_syn_train = copy.deepcopy(self.data.detach()), copy.deepcopy(self.targets.detach())
                optimizer_pretrained = torch.optim.SGD(pretrained_net.parameters(), lr=args.lr) 
                optimizer_pretrained.zero_grad()
                criterion = nn.CrossEntropyLoss().to(self.device)
                train_epoch(args, loader, pretrained_net,criterion, optimizer_pretrained, mixup=args.mixup)
                if isinstance(pretrained_net, torch.nn.DataParallel):
                    feature_syn_single = pretrained_net.module.get_features(image_syn_train, args.layer_idx).detach().clone()
                else:
                    feature_syn_single = pretrained_net.get_features(image_syn_train, args.layer_idx).detach().clone()
                # copy feature_syn into n_feat in the first dimension
                feature_syn.append(feature_syn_single)
                del pretrained_net
            feature_syn = torch.stack(feature_syn, dim=0).to(self.device)
            feature_syn.requires_grad_()
            self.features = feature_syn

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list
    def parameters_f(self):
        parameter_list = [self.features]
        return parameter_list

    def subsample(self, data, target,feat, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0) and (feat.shape[0] > max_size):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]
            feat = feat[:, indices[:max_size]]
        return data, target, feat

    def decode_zoom(self, img, target, feat, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        h_f = feat.shape[-1]
        remained = h % factor
        remained_f = h_f % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        if remained_f > 0:
            feat = F.pad(feat, pad=(0, factor - remained_f, 0, factor - remained_f), value=0.5)

        # print('fun(decode_zoom)  img feat',img.shape,feat.shape)

        s_crop = ceil(h / factor)
        n_crop = factor**2
        s_crop_f = ceil(h_f / factor)

        cropped = []
        cropped_f = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
                h_loc_f = i * s_crop_f
                w_loc_f = j * s_crop_f              
                cropped_f.append(feat[:, :, :, h_loc_f:h_loc_f + s_crop_f, w_loc_f:w_loc_f + s_crop_f])
        cropped = torch.cat(cropped)
        cropped_f = torch.cat(cropped_f,dim=1)

        # print('fun(decode_zoom)  cropped cropped_f',cropped.shape,cropped_f.shape)
        data_dec = self.resize(cropped)

        resized_features = []

        for i in range(cropped_f.shape[0]):
            resized_feature = self.resize_f(cropped_f[i])  
            resized_features.append(resized_feature)

        feat_dec = torch.stack(resized_features)

        

        # feat_dec = self.resize_f(cropped_f)
        target_dec = torch.cat([target for _ in range(n_crop)])
        # print('after resize feat_dec',feat_dec.shape)

        return data_dec, target_dec, feat_dec

    def decode_zoom_multi(self, img, target, feat, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        feature_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, feat, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])
            feature_multi.append(decoded[2])

        return torch.cat(data_multi), torch.cat(target_multi), torch.cat(feature_multi)

    # def decode_zoom_bound(self, img, target, factor_max, bound=128):
    #     """Uniform multi-formation with bounded number of synthetic data
    #     """
    #     bound_cur = bound - len(img)
    #     budget = len(img)

    #     data_multi = []
    #     target_multi = []

    #     idx = 0
    #     decoded_total = 0
    #     for factor in range(factor_max, 0, -1):
    #         decode_size = factor**2
    #         if factor > 1:
    #             n = min(bound_cur // decode_size, budget)
    #         else:
    #             n = budget

    #         decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
    #         data_multi.append(decoded[0])
    #         target_multi.append(decoded[1])

    #         idx += n
    #         budget -= n
    #         decoded_total += n * decode_size
    #         bound_cur = bound - decoded_total - budget

    #         if budget == 0:
    #             break

    #     data_multi = torch.cat(data_multi)
    #     target_multi = torch.cat(target_multi)

    #     return data_multi, target_multi

    def decode(self, data, target,feature, bound=128):
        """Multi-formation
        """
        if self.decode_type == 'multi':
            data, target, feat = self.decode_zoom_multi(data, target, feature, self.factor)
        elif self.decode_type == 'bound':
            data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
        else:
            data, target, feat = self.decode_zoom(data, target,feature,  self.factor)

        return data, target, feat

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]
        feat = self.features[:, idx_from:idx_to]

        data, target, feat = self.decode(data, target,feat, bound=max_size)
        data, target, feat = self.subsample(data, target,feat, max_size=max_size)
        return data, target, feat

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)

        
        # data, target,feature = self.decode(self.data.detach(), self.targets.detach(),self.features.detach())
        
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        feature_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            feature = self.features[:,idx_from:idx_to].detach()
            # data, target, feature = self.decode(data, target,feature)

            data_dec.append(data)
            target_dec.append(target)
            feature_dec.append(feature)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)
        feature_dec = torch.cat(feature_dec,dim=1)
        train_dataset = TensorDataset(data.cpu(), target.cpu(),feature.cpu(), train_transform)

        print("Decode condensed data: ", data.shape,feature.shape)
        nw = 0 if not augment else args.workers

        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)

        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        # Test on current model
        test_data(args, loader, val_loader, test_resnet=False, logger=logger)

        if bench:
            # Test on resnet-10 models
            test_data(args, loader, val_loader, test_resnet=True, logger=logger)

    def test_with_features(self, args, val_loader, feature_criterion, logger, bench=True,pooling_function=None,feat_loss=None,feature_strategy=None,pooling=None):
        """Condensed data evaluation
        """
        loader = self.loader(args, args.augment)
        feature_syn_eval = copy.deepcopy(self.features.detach())
        # print('========================================================')
        # print('fun(test_with_features)')
        # print('feature_syn_eval',feature_syn_eval.shape)
        # Test on current model
        test_data_with_features(args, loader, val_loader,feature_criterion,  test_resnet=False, logger=logger,pooling_function=pooling_function,feat_loss=feat_loss,feature_strategy=feature_strategy,pooling=args.poolFlag)

        if bench:
            # Test on resnet-10 models
            test_data_with_features(args, loader, val_loader,feature_criterion,  test_resnet=True, logger=logger,pooling_function=pooling_function,feat_loss=feat_loss,feature_strategy=feature_strategy,pooling=args.poolFlag)


def load_resized_data(args, subclass_list):
    """Load original training data (without augmentation) for condensation
    """
    if args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10    

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = resize
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,   
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

    # Validate only for the given subclass
    indices = []
    for i, c in enumerate(val_dataset.targets):
        if c in subclass_list:
            indices.append(i)

    val_dataset = torch.utils.data.Subset(val_dataset, indices)
    val_loader = MultiEpochsDataLoaderVal(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check
    return train_dataset, val_loader


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model):
    """Matching losses (feature or gradient)
    """
    loss = None

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))

    elif args.match == 'grad':
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

    return loss

def matchloss_w_feat(args, img_real, img_syn, lab_real, lab_syn, model,net_parameters_wo_classifier,feature, feat_syn,feature_criterion,acc_feat):
    """Matching losses (feature or gradient)
    """
    loss = None

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))

    elif args.match == 'grad':
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, net_parameters_wo_classifier)
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn) 
        # print('feature,feat_syn',feature.shape, feat_syn.shape)
        # feature,feat_syn torch.Size([40, 128, 4, 4]) torch.Size([1, 40, 128, 4, 4])
        # feature,feat_syn torch.Size([40, 128, 4, 4]) torch.Size([40, 128, 4, 4])
        loss_syn += args.lbd * feature_criterion(feature, feat_syn)

        if isinstance(model, torch.nn.DataParallel):
            output_feat_syn = model.module.get_output_from_features(feat_syn, layer_idx=args.layer_idx)
        else:
            output_feat_syn = model.get_output_from_features(feat_syn, layer_idx=args.layer_idx)
        loss_feat_cls = criterion(output_feat_syn, lab_syn)
        with torch.no_grad():
            acc_feat += torch.sum(torch.argmax(output_feat_syn.detach().clone(), dim=1) == lab_syn).item()

        g_syn = torch.autograd.grad(loss_syn, net_parameters_wo_classifier, create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not args.fc:
                continue

            loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))

    return loss,loss_feat_cls


def pretrain_sample(args, model, verbose=False):
    """Load pretrained networks
    """
    folder_base = f'./pretrained/{args.datatag}/{args.modeltag}_cut'
    folder_list = glob.glob(f'{folder_base}*')
    tag = np.random.randint(len(folder_list))
    folder = folder_list[tag]

    epoch = args.pt_from
    if args.pt_num > 1:
        epoch = np.random.randint(args.pt_from, args.pt_from + args.pt_num)
    ckpt = f'checkpoint{epoch}.pth.tar'

    file_dir = os.path.join(folder, ckpt)
    load_ckpt(model, file_dir, verbose=verbose)


def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    feature_strategy = args.use_feature
    print('feature_strategy: ', feature_strategy)
    feat_size = [128, 4, 4]
    if args.layer_idx:
        if args.layer_idx == 1:
            feat_size = [128,16, 16]
        elif args.layer_idx == 2:
            feat_size = [128, 8, 8]
    if args.pooling:
        feat_size = [128, 1, 1]

    nch_f, hs_f, ws_f = feat_size
    # Define subclass list and reverse mapping
    cls_from = args.phase * args.nclass_sub
    cls_to = (args.phase + 1) * args.nclass_sub
    subclass_list = np.arange(cls_from, cls_to)
    real_to_idx = {}
    for i, c in enumerate(subclass_list):
        real_to_idx[c] = i

    # Define real dataset and loader
    trainset, val_loader = load_resized_data(args, subclass_list)
    ## This load target subclass data on GPU memory
    ## One can modify ClassPartMemDataLoader to dynamically read subclass data without pre-loading
    ## Note, the entire class data is random sampled via the dataloader iterator (used for training networks)
    loader_real = ClassPartMemDataLoader(subclass_list,
                                         real_to_idx,
                                         trainset,
                                         batch_size=args.batch_real,
                                         num_workers=args.workers,
                                         shuffle=True,
                                         pin_memory=True,
                                         drop_last=True)

    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    # Define syn dataset
    synset = SynthesizerFeature(args, args.nclass_sub, subclass_list, nch, nch_f,hs, ws,hs_f,ws_f,args.n_feat)
    synset.init(args, loader_real, init_type=args.init)
    save_img(os.path.join(args.save_dir, 'init.png'),
             synset.data,
             unnormalize=False,
             dataname=args.dataset)

    # Define augmentation function
    aug, aug_rand = diffaug(args)
    save_img(os.path.join(args.save_dir, f'aug.png'),
             aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
             unnormalize=True,
             dataname=args.dataset)
        # 
    feature_syn = synset.features

    

    if args.contrast:
        adversarial_criterion = SupConLoss()


    if args.feat_metric == 'MSE':
        feature_criterion = nn.MSELoss()
    elif args.feat_metric == 'MMD':
        feature_criterion = MMDLoss()
    elif args.feat_metric == 'cosine':
        feature_criterion = CosineLoss()
    # if args.contrast:
    #     adversarial_criterion = SupConLoss()

    # define pooling function
    pooling_function = None
    images_norm, features_norm = [],[]
    if args.pooling:
        if args.pooling == 'avg':
            # avgpooling2d function
            pooling_function = partial(torch.nn.functional.adaptive_avg_pool2d, output_size=(1, 1))
        elif args.pooling == 'max':
            # maxpooling2d function
            pooling_function = partial(torch.nn.functional.adaptive_max_pool2d, output_size=(1, 1))
        elif args.pooling == 'sum':
            # sumpooling2d function
            pooling_function = partial(torch.nn.functional.adaptive_sum_pool2d, output_size=(1, 1))
        else:
            raise NotImplementedError('Pooling method not implemented')
    # 

    feature_criterion = feature_criterion.to(args.device)

    if not args.test:
        # synset.test(args, val_loader, logger, bench=False)

        synset.test_with_features(args, val_loader, feature_criterion,logger, bench=False,pooling_function=pooling_function,feat_loss=None,feature_strategy=feature_strategy,pooling=args.poolFlag)

    # Data distillation
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)
    # add synthetic features optimizer
    if args.feat_opt == 'SGD':
        optimizer_feat = torch.optim.SGD([feature_syn, ], lr=args.lr_feat, momentum=0.5) # optimizer_feat for synthetic features
    else:
        optimizer_feat = torch.optim.Adam([feature_syn, ], lr=args.lr_feat)
    optimizer_feat.zero_grad()

    

    ts = utils.TimeStamp(args.time)
    n_iter = args.niter * 100 // args.inner_loop
    it_log = n_iter // 50
    it_test = [n_iter // 10, n_iter // 5, n_iter // 2, n_iter]
    logger(f"\nStart condensing with {args.match} matching for {n_iter} iteration")
    args.fix_iter = max(1, args.fix_iter)
    for it in range(n_iter): 
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(it)
        print(current_time)
        if it % args.fix_iter == 0: 
            model = define_model(args, nclass).to(device)
            net_parameters_wo_classifier = [param for name, param in model.named_parameters() if 'classifier' not in name]
            model.train()
            optim_net = optim.SGD(model.parameters(),
                                  args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()

            if args.pt_from >= 0:
                pretrain_sample(args, model)

            if args.early > 0:
                for _ in range(args.early):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optim_net,
                                aug=aug_rand,
                                mixup=args.mixup_net)

        loss_total = 0
        loss_nfeat_total = 0
        loss_avg_feature_matching = np.zeros(args.n_feat)
        loss_avg_feature_cls = np.zeros(args.n_feat)
        acc_avg_feat = np.zeros(args.n_feat)
        synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        synset.features.data = torch.clamp(synset.features.data, min=0., max=1.)
        for ot in range(args.inner_loop):
            ts.set()
            # Update synset
            for c in subclass_list:
                img, lab = loader_real.class_sample(c)
                img_syn, lab_syn, _  = synset.sample(real_to_idx[c], max_size=args.batch_syn_max)
                ts.stamp("data")

                n = img.shape[0]
                img_aug = aug(torch.cat([img, img_syn]))
                ts.stamp("aug")

                loss = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model)
                loss_total += loss.item()
                ts.stamp("loss")

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()
                ts.stamp("backward")
            img_real_raw = [loader_real.class_sample(c)[0] for c in range(args.nclass_sub)] # TODO: fix here
            # TODO: Update feature synset
            for feat_idx in range(args.n_feat):
                optimizer_feat.zero_grad()
                loss_c_total = torch.tensor(0.0).to(args.device)
                acc_feat = 0
                for c in subclass_list:
                    img = img_real_raw[c].clone()
                    lab = loader_real.class_sample(c)[1]
                    img_syn, lab_syn, feat_syn  = synset.sample(real_to_idx[c], max_size=args.batch_syn_max)
                    feat_syn = feat_syn.squeeze(0)
                    img_syn = img_syn.detach().clone()
                    # print('==========================================================')
                    # print('img,lab,img_syn,lab_syn feat_syn ',img.shape,lab.shape,img_syn.shape,lab_syn.shape,feat_syn.shape)
                    # torch.Size([64, 3, 32, 32]) torch.Size([64]) torch.Size([40, 3, 32, 32]) torch.Size([40]) torch.Size([40, 128, 4, 4])
                    # feat_syn = synset.sample(c, max_size=args.batch_syn_max)

                    # print('feature_syn[feat_idx, c*args.ipc:(c+1)*args.ipc]  feat_syn ',feature_syn[feat_idx, c*args.ipc:(c+1)*args.ipc].shape,feat_syn.shape)
                    ts.stamp("data")

                    n = img.shape[0]
                    img_aug = aug(torch.cat([img, img_syn]))
                    ts.stamp("aug")

                    if isinstance(model, torch.nn.DataParallel):
                        feature = model.module.get_features(img_syn, args.layer_idx)
                    else:
                        feature = model.get_features(img_syn, args.layer_idx)
                    if args.pooling:
                        feature = pooling_function(feature)
                    if args.feat_norm:
                        feature = torch.nn.functional.normalize(feature, p=2, dim=1)
                        feat_syn = torch.nn.functional.normalize(feat_syn, p=2, dim=1)


                    # print('feature feat_syn',feature.shape,feat_syn.shape)
                    # feature,feat_syn torch.Size([40, 128, 4, 4]) torch.Size([1, 40, 128, 4, 4])
                    # feature,feat_syn torch.Size([40, 128, 4, 4]) torch.Size([40, 128, 4, 4])
                    loss, loss_feat_cls = matchloss_w_feat(args, img_aug[:n], img_aug[n:], lab, lab_syn, model, net_parameters_wo_classifier,feature,feat_syn,feature_criterion,acc_feat)
                    
                    # ipdb.set_trace()
                    # feature_temp=torch.nn.functional.normalize(feature_syn[feat_idx])
                    loss += loss_feat_cls * args.feat_lbd

                    loss_c_total += loss.item()
                    ts.stamp("loss")

                    optimizer_feat.zero_grad()
                    loss.backward()
                    optimizer_feat.step()
                    ts.stamp("backward")
                acc_avg_feat[feat_idx] += acc_feat
                
                loss_avg_feature_matching[feat_idx] += loss.item()
                loss_avg_feature_cls[feat_idx] += loss_feat_cls.item()
            # Net update
            if args.n_data > 0:
                trainset, val_loader = load_resized_data(args)
                loader_net = loader_real
                pretrained_net = define_model(args, args.nclass_sub).to(args.device) # get a random model
                pretrained_net.train()
                image_syn_train, label_syn_train = copy.deepcopy(synset.data.detach()), copy.deepcopy(synset.targets.detach())
                optimizer_pretrained = torch.optim.SGD(pretrained_net.parameters(), lr=args.lr) 
                optimizer_pretrained.zero_grad()
                criterion = nn.CrossEntropyLoss().to(args.device)
                train_epoch(args, loader_net, pretrained_net,criterion, optimizer_pretrained, n_data=args.n_data,
                                aug=aug_rand,
                                mixup=args.mixup_net)

                loader = synset.loader(args, args.augment)
                for _ in range(args.net_epoch):
                    train_epoch_with_features(args,
                                loader,
                                model,
                                criterion,
                                feature_criterion,
                                optim_net,
                                n_data=args.n_data,
                                aug=aug_rand,
                                mixup=args.mixup_net,
                                lbd=args.lbd,
                                pooling_function=pooling_function,
                                layer_idx=args.layer_idx, 
                                feat_loss=loss_avg_feature_cls, 
                                feature_strategy=feature_strategy,feat_flag=args.feat_flag,pretrained_net=pretrained_net,n_feat=args.n_feat,pooling=args.poolFlag)
            ts.stamp("net update")

            if (ot + 1) % 10 == 0:
                ts.flush()

        # Logging
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total/args.nclass_sub/args.inner_loop:.1f}"
            )
        if (it + 1) in it_test:
            save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                     synset.data,
                     unnormalize=False,
                     dataname=args.dataset)
            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data.pt'))
            print("img and data saved!")

            if not args.test:
                # synset.test(args, val_loader, logger)
                print('Evaluate without feature!')
                # synset.test_with_features(args, val_loader, logger)
                synset.test_with_features(args, val_loader, feature_criterion,logger, pooling_function=pooling_function,feat_loss=feat_loss,feature_strategy=feature_strategy,pooling=args.poolFlag)


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    cudnn.benchmark = True

    assert args.ipc > 0
    if args.nclass_sub < 0:
        raise AssertionError("Set number of subclasses! (args.nclass_sub)")
    if args.phase < 0:
        raise AssertionError("Set phase number! (args.phase)")

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    condense(args, logger)