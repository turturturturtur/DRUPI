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
    # if args.contrast:
    #     model_args += f'_contrast{args.contrastType}{args.lbd_contrast}_{args.SupConfType}'
    # else:
    #     model_args += f'_contrast{args.contrast}'
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
           
            
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key] # accs: (num_exp, num_eval)
        accs = accs.reshape(-1)
        print('Run %d experiments with Strategy %s, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, feature_strategy,args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))
if __name__ == '__main__':
    main()
