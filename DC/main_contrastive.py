import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDatasetFeature, TensorDataset, epoch, epoch_with_features, DiffAugment, ParamDiffAug, FlushFile, evaluate_synset_w_feature
import sys
from functools import partial
import matplotlib.pyplot as plt
from feature_metric import MMDLoss, CosineLoss, SupConLoss , compute_info_nce_loss
import os.path as osp
import matplotlib.pyplot as plt
# import os

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/cpfs01/user/luanqi.p/wangshaobo/data/OpenDataLab___CIFAR-10/raw', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--lbd', type=float, default=0.01, help='scale of MSE')
    parser.add_argument('--lr_feat', type=float, default=0.1, help='learning rate for updating synthetic features')
    parser.add_argument('--layer-idx', type=int, default=None, help='layer index for feature')
    parser.add_argument('--pooling', type=str, default=None, help='feature pooling method')
    parser.add_argument('--feat-lbd', type=float, default=0, help='scale of CE')
    parser.add_argument('--feat-opt', type=str, default='SGD', help='featureoptimizer')
    parser.add_argument('--n-feat', type=int, default=1, help='number of features')
    parser.add_argument('--eval-freq', type=int, default=4, help='evaluate frequency')
    parser.add_argument('--use-feature', type=str, default='mean', help='the way to use the feature')
    parser.add_argument('--feat-norm', action='store_true', default=False, help='normalize the feature')
    parser.add_argument('--contrast', action='store_true', default=False, help='use the contrastive loss')
    parser.add_argument('--poolFlag', action='store_true', default=False, help='use average pooling')
    parser.add_argument('--imgUdp', action='store_true', default=False, help='load img update')
    parser.add_argument('--contrastType', type=str, default='SupCon', help='the Type of Contrast Loss')
    parser.add_argument('--feat_flag', action='store_true', default=False, help='the source of using feature_syn in the function,default is False means from dataloader,true means from pretrained_model')
    parser.add_argument('--lbd-contrast', type=float, default=0.1, help='scale of contrastive loss')
    parser.add_argument('--lbd_ce', type=float, default=0.1, help='scale of contrastive loss')
    parser.add_argument('--feat-metric', type=str, default='MSE', help='feature criterion')
    parser.add_argument('--img-method',default='DC',help='image distillation method')
    parser.add_argument('--img-path', default='../distilled_data', help='image path')
    parser.add_argument('--SupConfType', type=str, default='all', help='SupConf Loss Type')
    args = parser.parse_args()
    final_acc=[]
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    data_path = os.path.expanduser(args.data_path)
    args.save_path = os.path.join(args.save_path, args.dataset, f'IPC{args.ipc}')
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)


    model_args = f'{args.dataset}_{args.model}_{args.ipc}ipc_[MSE_{args.lbd}_pool{args.pooling}_layer{args.layer_idx}_CE{args.feat_lbd}_nfeat{args.n_feat}_use{args.use_feature}_norm{args.feat_norm}]_{args.img_method}_{get_time()}_poolingFlag{args.poolFlag}_imgUdp{args.imgUdp}'
    if args.contrast:
        model_args += f'_contrast{args.contrastType}{args.lbd_contrast}_{args.SupConfType}'
    else:
        model_args += f'_contrast{args.contrast}'
    if args.method=='DSA':
        model_args += f'_DSA_{args.dsa_strategy}'
    else:
        pass

    log_file = os.path.join(args.save_path, f'log_{model_args}.txt')
    load_path = osp.join(args.img_path, args.img_method, args.dataset, f'IPC{args.ipc}')

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
        
    log_file_handle = open(log_file, 'w')
    sys.stdout = FlushFile(log_file_handle)
    eval_it_pool = np.arange(0, args.Iteration+1, args.Iteration//args.eval_freq).tolist()
    # eval_it_pool = np.arange(0, args.Iteration+1, args.Iteration//args.eval_freq).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    if args.feat_metric == 'MSE':
        feature_criterion = nn.MSELoss()
    elif args.feat_metric == 'MMD':
        feature_criterion = MMDLoss()
    elif args.feat_metric == 'cosine':
        feature_criterion = CosineLoss()
    if args.contrast:
        adversarial_criterion = SupConLoss()
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = np.zeros((args.num_exp, args.num_eval))
    data_save = []
    loss_img_match=[]
    loss_feat_all=[]
    loss_feat_match=[]
    loss_mse=[]
    loss_it=[]
    loss_feat_ce=[]
    ''' load the pretrained synthetic data '''
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
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
        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
        ''' initialize the synthetic data '''
        label_syn = torch.arange(num_classes, device=args.device).repeat_interleave(args.ipc).long()
        if args.init == 'real':
            print('initialize synthetic data from pretrained data')
            image_syn_pretrained = torch.load(osp.join(load_path, f'images_best.pt')).to(args.device)
            if args.imgUdp:
                image_syn = torch.tensor(image_syn_pretrained.clone(), dtype=torch.float, requires_grad=True, device=args.device)
            else:
                image_syn = torch.tensor(image_syn_pretrained.clone(), dtype=torch.float, requires_grad=False, device=args.device)
            del image_syn_pretrained
            feature_syn = []
            for _ in range(args.n_feat):
                pretrained_net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
                pretrained_net.train()
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                image_train, label_train = copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_train, label_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                optimizer_pretrained = torch.optim.SGD(pretrained_net.parameters(), lr=args.lr_net) 
                optimizer_pretrained.zero_grad()
                criterion = nn.CrossEntropyLoss().to(args.device)
                epoch('train', trainloader, pretrained_net, optimizer_pretrained, criterion, args, False)
                if isinstance(pretrained_net, torch.nn.DataParallel):
                    feature_syn_single = pretrained_net.module.get_features(image_syn_train, args.layer_idx).detach().clone()
                else:
                    feature_syn_single = pretrained_net.get_features(image_syn_train, args.layer_idx).detach().clone()
                # copy feature_syn into n_feat in the first dimension
                # print('feature_syn_single.shape', feature_syn_single.shape)
                feature_syn.append(feature_syn_single)
                del pretrained_net
            feature_syn = torch.stack(feature_syn, dim=0).to(args.device)
            if args.pooling:
                feature_syn = pooling_function(feature_syn)
            feature_syn.requires_grad_()
        else:
            print('initialize synthetic data from random noise')
            image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
            feature_syn = torch.randn(size=(args.n_feat, num_classes*args.ipc, feat_size[0], feat_size[1], feat_size[2]), dtype=torch.float, requires_grad=True, device=args.device)
        ''' training '''
        
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic image
        optimizer_img.zero_grad()
        # add synthetic features optimizer
        if args.feat_opt == 'SGD':
            optimizer_feat = torch.optim.SGD([feature_syn, ], lr=args.lr_feat, momentum=0.5) # optimizer_feat for synthetic features
        else:
            optimizer_feat = torch.optim.Adam([feature_syn, ], lr=args.lr_feat)
        optimizer_feat.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        feature_criterion = feature_criterion.to(args.device)
        print('%s training begins'%get_time())
        for it in range(args.Iteration+1):
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)
                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300
                    accs = np.zeros((args.num_eval))
                    if it == 0:
                        feat_loss = None
                    else:
                        feat_loss = loss_avg_feature_cls
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        feature_syn_eval = copy.deepcopy(feature_syn.detach())
                        if it_eval == 0:
                            print('Evaluate without feature!')
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        else:
                            _, acc_train, acc_test = evaluate_synset_w_feature(it_eval, net_eval, image_syn_eval, label_syn_eval, feature_syn_eval, testloader, args, feature_criterion, pooling_function=pooling_function,feat_loss=feat_loss,feature_strategy=feature_strategy,pooling=args.poolFlag)
                        accs[it_eval] = acc_test
                    print('Evaluate %d random %s, mean = %.4f std = %.4f strategy = %s \n-------------------------'%(accs.shape[0], model_eval, np.mean(accs), np.std(accs), feature_strategy))
                    # final_acc.append(np.mean(accs))
                    # if it == args.Iteration: # record the final results
                    #     accs_all_exps[model_eval][exp] =  accs
                    
                    if np.mean(accs) > np.mean(accs_all_exps[model_eval][exp]):
                        accs_all_exps[model_eval][exp] = accs
                ''' visualize and save '''
                # add pooling function name if it is not none
                save_name = os.path.join(args.save_path, f'vis_{model_args}_exp{exp}_iter{it}.png')
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu()), copy.deepcopy(feature_syn.detach().cpu())])
                model_path =  os.path.join(args.save_path, f'__Iter{it}__model_{model_args}.pt')
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, model_path)
            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            net_parameters_wo_classifier = [param for name, param in net.named_parameters() if 'classifier' not in name]
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net) 
            optimizer_net.zero_grad()
            loss_avg = 0
            loss_avg_feature_matching = np.zeros(args.n_feat)
            loss_avg_feature_cls = np.zeros(args.n_feat)
            acc_avg_feat = np.zeros(args.n_feat)
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.
            for ol in range(args.outer_loop):
                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.
                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer
                ''' update synthetic image '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))                    
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    output_real = net(img_real)
                    # grad of CE loss of img_real
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    # grad of CE loss of img_syn
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn) 
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss += match_loss(gw_syn, gw_real, args)
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()
                img_real_raw = [get_images(c, args.batch_real) for c in range(num_classes)] # TODO: fix here
                ''' update synthetic features '''
                for feat_idx in range(args.n_feat):
                    optimizer_feat.zero_grad()
                    loss = torch.tensor(0.0).to(args.device)
                    acc_feat = 0
                    for c in range(num_classes):
                        img_real = img_real_raw[c].clone() # TODO: fix the img_real
                        lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                        img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                        img_syn = img_syn.detach().clone() # TODO: add here
                        lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                        # The code `feat_syn` is a placeholder or a comment in Python. It does not
                        # have any functionality and is typically used to indicate a feature or a
                        # section of code that is planned but not yet implemented.
                        # print('feature_syn',feature_syn.shape)
                        feat_syn = feature_syn[feat_idx, c*args.ipc:(c+1)*args.ipc].reshape((args.ipc,  feat_size[0], feat_size[1], feat_size[2])) 
                        # print('feat_syn',feat_syn.shape)   
                        # print(feat_syn.requires_grad)
                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        # grad of CE loss of img_real
                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters_wo_classifier)
                        gw_real = list((_.detach().clone() for _ in gw_real))
                        if isinstance(net, torch.nn.DataParallel):
                            feature = net.module.get_features(img_syn, args.layer_idx)
                        else:
                            feature = net.get_features(img_syn, args.layer_idx)
                        if args.pooling:
                            feature = pooling_function(feature)
                        if args.feat_norm:
                            feature = torch.nn.functional.normalize(feature, p=2, dim=1)
                            feat_syn = torch.nn.functional.normalize(feat_syn, p=2, dim=1)
                        # grad of CE loss of img_syn and grad of MSE loss of feat_syn

                        # print('feature_after',feature.shape,feat_syn.shape)
                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn) + args.lbd * feature_criterion(feature, feat_syn)
                        
                        # print(img_syn.shape,feat_syn.shape)
                        if isinstance(net, torch.nn.DataParallel):
                            output_feat_syn = net.module.get_output_from_features(feat_syn, layer_idx=args.layer_idx)
                            # output_feat_syn = net.module.classifier(feat_syn.view(feat_syn.size(0), -1))
                        else:
                            output_feat_syn = net.get_output_from_features(feat_syn, layer_idx=args.layer_idx)
                            # output_feat_syn = net.classifier(feat_syn.view(feat_syn.size(0), -1))
                        loss_feat_cls = criterion(output_feat_syn, lab_syn)
                        with torch.no_grad():
                            acc_feat += torch.sum(torch.argmax(output_feat_syn.detach().clone(), dim=1) == lab_syn).item()
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters_wo_classifier, create_graph=True)
                        loss += match_loss(gw_syn, gw_real, args)
                        # print("match loss",match_res)

                        feature_temp=torch.nn.functional.normalize(feature_syn[feat_idx])
                        loss_ce=criterion(feature,feat_syn)
                        loss += loss_feat_cls * args.feat_lbd
                        # loss +=loss_ce*args.lbd_ce
                        if args.contrast:
                            '''add contrastive loss'''
                            if args.contrastType == "InfoNCE":
                                loss_adv = compute_info_nce_loss(feature_temp, label_syn, c)
                                # print("loss_ad
                                # v:",loss_adv)
                            elif args.contrastType =="SupCon":
                                sup_con_loss = SupConLoss(temperature=0.07, contrast_mode=args.SupConfType)
                                loss_adv = sup_con_loss(feature_temp, label_syn)
                                # print("loss_adv:",loss_adv)
                            loss += loss_adv * args.lbd_contrast
                            # print("loss_feat_cls:",loss_feat_cls)
                            
                    acc_avg_feat[feat_idx] += acc_feat
                    optimizer_feat.zero_grad()
                    loss.backward()
                    optimizer_feat.step()
                    loss_avg_feature_matching[feat_idx] += loss.item()
                    loss_avg_feature_cls[feat_idx] += loss_feat_cls.item()
                    # loss_match
                if ol == args.outer_loop - 1:
                    break
                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                feat_syn_train = copy.deepcopy(feature_syn.detach()) # TODO: modified here
                dst_syn_train = TensorDatasetFeature(image_syn_train, label_syn_train, feat_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                # =================================modify epoch with features feature_syn
                pretrained_net = get_network(args.model, channel, num_classes, im_size).to(args.device)
                pretrained_net.train()
                image_train, label_train = copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach())
                # avoid any unaware modification
                dst_syn_train = TensorDataset(image_train, label_train)
                trainloader_pre = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True,
                                                          num_workers=0)
                optimizer_pretrained = torch.optim.SGD(pretrained_net.parameters(), lr=args.lr_net)
                optimizer_pretrained.zero_grad()
                criterion = nn.CrossEntropyLoss().to(args.device)
                epoch('train', trainloader_pre, pretrained_net, optimizer_pretrained, criterion, args, False)

                for il in range(args.inner_loop):

                    epoch_with_features('train', trainloader, net, optimizer_net, criterion, feature_criterion, args, aug = True if args.dsa else False, 
                                        lbd=args.lbd,
                                        pooling_function=pooling_function,
                                        layer_idx=args.layer_idx, 
                                        feat_loss=loss_avg_feature_cls, 
                                        feature_strategy=feature_strategy,feat_flag=args.feat_flag,pretrained_net=pretrained_net,n_feat=args.n_feat,pooling=args.poolFlag)

            loss_avg /= (num_classes*args.outer_loop)
            loss_avg_feature_matching /= (num_classes*args.outer_loop)

            loss_avg_feature_cls /= (num_classes*args.outer_loop)
            acc_avg_feat /= (num_classes*args.outer_loop*args.ipc)
            acc_avg_feat = f'{acc_avg_feat*100}%'
            if it % 50 == 0:
                print('%s iter = %04d, Matching loss = %.4f' % (get_time(), it, loss_avg))
                print('feat matching loss = ', loss_avg_feature_matching)
                print('feat cls loss = ', loss_avg_feature_cls)
                print('feat cls acc = ', acc_avg_feat)
                loss_img_match.append(loss_avg)
                loss_feat_all.append(loss_avg_feature_matching)
                loss_feat_match.append(loss_avg_feature_matching-loss_avg_feature_cls * args.feat_lbd)
                loss_feat_ce.append(loss_avg_feature_cls)
                loss_it.append(it)
                

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu()), copy.deepcopy(feature_syn.detach().cpu())])
                model_path =  os.path.join(args.save_path, f'model_{model_args}.pt')
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, model_path)

                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                axs[0, 0].plot(loss_it, loss_img_match, color='blue')
                axs[0, 0].set_title('Image Matching Loss')
                axs[0, 0].set_xlabel('Iterations')
                axs[0, 0].set_ylabel('Loss')

                axs[0, 1].plot(loss_it, loss_feat_all, color='green')
                axs[0, 1].set_title('Feature All Loss')
                axs[0, 1].set_xlabel('Iterations')
                axs[0, 1].set_ylabel('Loss')

                axs[1, 0].plot(loss_it, loss_feat_match, color='red')
                axs[1, 0].set_title('Feature Matching Loss')
                axs[1, 0].set_xlabel('Iterations')
                axs[1, 0].set_ylabel('Loss')

                axs[1, 1].plot(loss_it, loss_feat_ce, color='orange')
                axs[1, 1].set_title('Feature Classification Loss')
                axs[1, 1].set_xlabel('Iterations')
                axs[1, 1].set_ylabel('Loss')

                plt.tight_layout()

                save_path = os.path.join(args.save_path, f'loss_curves_subplot__model__{model_args}.png')

                plt.savefig(save_path)
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key] # accs: (num_exp, num_eval)
        accs = accs.reshape(-1)
        print('Run %d experiments with Strategy %s, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, feature_strategy,args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
if __name__ == '__main__':
    main()
