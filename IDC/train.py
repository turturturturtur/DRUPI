# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN
from data import load_data, MEANS, STDS
from misc.utils import random_indices, rand_bbox, AverageMeter, accuracy, get_time, Plotter
from misc.augment import DiffAug
from efficientnet_pytorch import EfficientNet
import time
import warnings
import ipdb

warnings.filterwarnings("ignore")
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

mean_torch = {}
std_torch = {}
for key, val in MEANS.items():
    mean_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)
for key, val in STDS.items():
    std_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)


class ConvNet(nn.Module):
    def __init__(self, nclass, net_norm, net_depth, net_width, channel, im_size):
        super(ConvNet, self).__init__()
        
        layers = []
        in_channels = channel
        
        # Example of constructing layers
        for _ in range(net_depth):
            layers.append(nn.Conv2d(in_channels, net_width, kernel_size=3, stride=1, padding=1))
            
            if net_norm == 'batch':
                layers.append(nn.BatchNorm2d(net_width))
            elif net_norm == 'layer':
                layers.append(nn.LayerNorm([net_width, im_size[0], im_size[1]]))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = net_width
            im_size = (im_size[0] // 2, im_size[1] // 2)
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels * im_size[0] * im_size[1], nclass)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x, layer_idx=None):
        if layer_idx is None:
            return self.features(x)
        out = x
        tgt_idx = 0
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                tgt_idx += 1
            out = layer(out)
            if tgt_idx == layer_idx:
                return out
        return out

    def get_output_from_features(self, f, layer_idx=None):
        if layer_idx == None:
            f = f.view(f.shape[0], -1)
            return self.classifier(f)
        out = f
        tgt_idx = 0
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                tgt_idx += 1
            if tgt_idx > layer_idx:
                out = layer(out)
        out = out.view(out.shape[0],-1)
        out = self.classifier(out)
        return out

def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """
    if size == None:
        size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset,
                          args.depth,
                          nclass,
                          norm_type=args.norm_type,
                          size=size,
                          nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.dataset,
                              args.depth,
                              nclass,
                              width=args.width,
                              norm_type=args.norm_type,
                              size=size,
                              nch=args.nch)
    elif args.net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        width = int(128 * args.width)
        # model = CN.ConvNet(nclass,
        #                    net_norm=args.norm_type,
        #                    net_depth=args.depth,
        #                    net_width=width,
        #                    channel=args.nch,
        #                    im_size=(args.size, args.size))
        model = ConvNet(nclass=args.nclass,
                net_norm=args.norm_type,
                net_depth=args.depth,
                net_width=width,
                channel=args.nch,
                im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")
        # logger('# model parameters: {:.1f}M'.format(
        #     sum([p.data.nelement() for p in model.parameters()]) / 10**6))

    return model


def main(args, logger, repeat=1):
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    _, train_loader, val_loader, nclass = load_data(args)

    best_acc_l = []
    acc_l = []
    for i in range(repeat):
        logger(f"\nRepeat: {i+1}/{repeat}")
        plotter = Plotter(args.save_dir, args.epochs, idx=i)
        model = define_model(args, nclass, logger)

        best_acc, acc = train(args, model, train_loader, val_loader, plotter, logger)
        best_acc_l.append(best_acc)
        acc_l.append(acc)

    logger(f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_l):.1f} {np.mean(acc_l):.1f}')


def train(args, model, train_loader, val_loader, plotter=None, logger=None):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    if args.pretrained:
        pretrained = "{}/{}".format(args.save_dir, 'checkpoint.pth.tar')
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)
        # TODO: optimizer scheduler steps

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    if args.dsa:
        aug = DiffAug(strategy=args.dsa_strategy, batch=False)
        logger(f"Start training with DSA and {args.mixup} mixup")
    else:
        aug = None
        logger(f"Start training with base augmentation and {args.mixup} mixup")

    # Start training and validation
    # print(get_time())
    for epoch in range(cur_epoch + 1, args.epochs + 1):
        # print("==========",args.epochs)
        acc1_tr, _, loss_tr = train_epoch(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          epoch,
                                          logger,
                                          aug,
                                          mixup=args.mixup)

        if epoch % args.epoch_print_freq == 0:
            # print("==========test")
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch, logger)

            if plotter != None:
                plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                if logger != None:
                    logger(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        if args.save_ckpt and (is_best or (epoch == args.epochs)):
            state = {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(args.save_dir, state, is_best)
        scheduler.step()

    return best_acc1, acc1


def train_with_features(args, model, train_loader, val_loader, criterion_feature,plotter=None, logger=None,pooling_function=None,
    feat_loss=None,
    feature_strategy=None,
    pooling=True,):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    if args.pretrained:
        pretrained = "{}/{}".format(args.save_dir, 'checkpoint.pth.tar')
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)
        # TODO: optimizer scheduler steps

    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    if args.dsa:
        aug = DiffAug(strategy=args.dsa_strategy, batch=False)
        logger(f"Start training with DSA and {args.mixup} mixup")
    else:
        aug = None
        logger(f"Start training with base augmentation and {args.mixup} mixup")

    # Start training and validation
    # print(get_time())
    for epoch in range(cur_epoch + 1, args.epochs + 1):
        acc1_tr, _, loss_tr = train_epoch_with_features(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          criterion_feature,
                                          optimizer,
                                          epoch,
                                          logger,
                                          aug,
                                          lbd=args.lbd,
                                        pooling_function=pooling_function,
                                        layer_idx=args.layer_idx,
                                        feat_loss=feat_loss,
                                        feature_strategy=feature_strategy,
                                        pooling=pooling,
                                          mixup=args.mixup)

        if epoch % args.epoch_print_freq == 0:
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch, logger)

            if plotter != None:
                plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                if logger != None:
                    logger(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        if args.save_ckpt and (is_best or (epoch == args.epochs)):
            state = {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(args.save_dir, state, is_best)
        scheduler.step()

    return best_acc1, acc1



def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch=0,
                logger=None,
                aug=None,
                mixup='vanilla',
                n_data=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    num_exp = 0
    for i, (input, target,_) in enumerate(train_loader):
        # for j, item in enumerate(batch):
        #     print(f"  Element {j}: type = {type(item)}, shape = {item.shape if hasattr(item, 'shape') else 'No shape'}")
        # input, target = batch
        if train_loader.device == 'cpu':
            input = input.cuda()
            target = target.cuda()

        data_time.update(time.time() - end)

        if aug != None:
            with torch.no_grad():
                input = aug(input)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(train_loader),
                                                                     batch_time=batch_time,
                                                                     data_time=data_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

        num_exp += len(target)
        if (n_data > 0) and (num_exp >= n_data):
            break

    if (epoch % args.epoch_print_freq == 0) and (logger is not None):
        logger(
            '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))

    return top1.avg, top5.avg, losses.avg


def train_epoch_with_features(args,
                train_loader,
                model,
                criterion,
                criterion_feature, 
                optimizer,
                epoch=0,
                logger=None,
                aug=None,
                lbd=1,
                pooling_function=None,
                layer_idx=None,
                feat_loss=None,
                feature_strategy=None,
                feat_flag=False,
                pretrained_net=None,
                n_feat=0,
                pooling=True,
                mixup='vanilla',
                n_data=-1,):
    # current_time_start = datetime.now()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    num_exp = 0

    if feat_loss is None:
        feat_loss = np.zeros(args.n_feat)

    criterion_feature = criterion_feature.to(args.device)

    for i, (input, target,feature_syn) in enumerate(train_loader):
        feature_syn=feature_syn.squeeze(0)
        # print("================")
        # print(i)
        # print("================")
    # for i, batch in enumerate(train_loader):
    # 打印整个 batch\

        # if i==0:
        #     print('================================')
        #     print('fun train_epoch_with_features')
        #     print('input, target,feature_syn',input.shape, target.shape,feature_syn.shape)
            # fun train_epoch_with_features
            # torch.Size([64, 3, 32, 32]) torch.Size([64]) torch.Size([1, 64, 128, 4, 4])

        if train_loader.device == 'cpu':
            input = input.cuda()
            target = target.cuda()
            feature_syn = feature_syn.cuda()

        data_time.update(time.time() - end)

        if aug != None:
            with torch.no_grad():
                input = aug(input)

        feature_syn = feature_syn.float().to(args.device)

        if feature_strategy == "mean":
            feature_syn = torch.mean(feature_syn, dim=0)  # use average feature
            
        elif feature_strategy == "random":
            feature_syn = feature_syn[
                :, torch.randint(0, feature_syn.shape[0], (1,)).item()
            ]

        # if i==0:
        #     print('after feature_strategy:feature_syn',feature_syn.shape)
            # after feature_strategy:feature_syn torch.Size([64, 128, 4, 4])

        # elif feature_strategy == "max":
        #     best_feature = np.argmin(feat_loss)
        #     feature_syn = feature_syn[:, best_feature]

        if isinstance(model, torch.nn.DataParallel):
            feature = model.module.get_features(input.clone(), layer_idx)
        else:
            feature = model.get_features(input.clone(), layer_idx)
        # if i==0:
        #     print('feature',feature.shape)
            # feature torch.Size([64, 128, 4, 4])
        if pooling_function:
            feature = pooling_function(feature)
            feature_syn = pooling_function(feature_syn)
        if args.feat_norm:
            feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            feature_syn = torch.nn.functional.normalize(feature_syn, p=2, dim=1)
        if feature.shape != feature_syn.shape:
            raise ValueError(f"Shape mismatch: feature shape {feature.shape}, feature_syn shape {feature_syn.shape}")

        r = np.random.rand(1)
        # if r < args.mix_p and mixup == 'cut':
        #     # generate mixed sample
        #     lam = np.random.beta(args.beta, args.beta)
        #     rand_index = random_indices(target, nclass=args.nclass)

        #     target_b = target[rand_index]
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        #     # bbx_f1, bby_f1, bbx_f2, bby_f2 = rand_bbox(feature_syn.size(), lam)

        #     input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        #     # print('input.shape',input.shape)
        #     # input.shape torch.Size([64, 3, 32, 32])
        #     # feature_syn[:, :, bbx_f1:bbx_f2, bby_f1:bby_f2] = feature_syn[rand_index, :, bbx_f1:bbx_f2, bby_f1:bby_f2]
        #     # print('feature_syn.shape',feature_syn.shape)
        #     # feature_syn.shape torch.Size([64, 128, 4, 4])
        #     ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

        #     output = model(input)
        #     loss_cls = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
        #     # loss_cls = loss
        # else:
        #     # compute output
        #     output = model(input)
        #     loss_cls = criterion(output, target)
            # loss_cls = loss

        output = model(input)
        loss_cls = criterion(output, target)
        # print("=============")
        # print("input",input.shape)

        # print("output",output.shape)
        # print("target",target)
        # print("loss_cls",loss_cls)
        # print("=============")
        

        if pooling:
            avg_pool = nn.AvgPool2d(kernel_size=4)
            feature=avg_pool(feature)
            feature_syn=avg_pool(feature_syn)
        else:
            pass
        # if feature.size(1)!=feature_syn.size(1):           
        #     fc1=nn.Linear(feature.size(1), feature_syn.size(1))
        #     feature = feature.permute(0, 2, 3, 1)
        #     feature=fc1(feature)
        #     feature = feature.permute(0, 3, 1, 2)

        loss_reg = criterion_feature(feature, feature_syn)
        # print("=============")
        # print("feature",feature.shape)
        # print("feature_syn",feature_syn.shape)
        # print("loss_reg",loss_reg)
        # print("=============")
        # loss = loss_cls + lbd * loss_reg
        loss = loss_cls
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print("==============")
        # print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time))
        # print("==============")
        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(train_loader),
                                                                     batch_time=batch_time,
                                                                     data_time=data_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

        num_exp += len(target)
        if (n_data > 0) and (num_exp >= n_data):
            break

    if (epoch % args.epoch_print_freq == 0) and (logger is not None):
        logger(
            '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    # current_time_start = datetime.now()
    
    # print()
    # ipdb.set_trace()
    return top1.avg, top5.avg, losses.avg


def validate(args, val_loader, model, criterion, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-acc {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-acc {top5.val:.4f} ({top5.avg:.4f})'.format(epoch,
                                                                     args.epochs,
                                                                     i,
                                                                     len(val_loader),
                                                                     batch_time=batch_time,
                                                                     loss=losses,
                                                                     top1=top1,
                                                                     top5=top5))

    if logger is not None:
        logger(
            '(Test ) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        checkpoint['state_dict'] = dict(
            (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'(epoch: {}, best acc1: {}%)".format(
            path, cur_epoch, checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
        cur_epoch = 0
        best_acc1 = 100

    return cur_epoch, best_acc1


def save_checkpoint(save_dir, state, is_best):
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    print("checkpoint saved! ", ckpt_path)


if __name__ == '__main__':
    from misc.utils import Logger
    from argument import args

    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")

    main(args, logger, args.repeat)
