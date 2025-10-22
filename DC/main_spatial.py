import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDatasetFeature, epoch_with_features, DiffAugment, ParamDiffAug, FlushFile
import sys
from functools import partial
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
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

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    data_path = os.path.expanduser(args.data_path)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model_args = f'{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc_[MSE{args.lbd}_pool{args.pooling}_layer{args.layer_idx}_CE{args.feat_lbd}_opt{args.feat_opt}_nfeat{args.n_feat}_use{args.use_feature}]'
    log_file = os.path.join(args.save_path, f'log_{model_args}.txt')

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

    eval_it_pool = np.arange(0, args.Iteration+1, args.Iteration//args.eval_freq).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    feature_criterion = nn.MSELoss()

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = np.zeros((args.num_exp, args.num_eval))

    data_save = []


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
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.arange(num_classes, device=args.device).repeat_interleave(args.ipc).long()
 
        feature_syn = torch.randn(size=(args.n_feat, num_classes*args.ipc, feat_size[0], feat_size[1], feat_size[2]), dtype=torch.float, requires_grad=True, device=args.device)
        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                idx_shuffle = np.random.permutation(indices_class[c])[:args.ipc]
                image_syn.data[c*args.ipc:(c+1)*args.ipc]= images_all[idx_shuffle]
        else:
            print('initialize synthetic data from random noise')

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
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, feature_syn_eval, testloader, args, pooling_function=pooling_function,feat_loss=feat_loss,feature_strategy=feature_strategy)
                        accs[it_eval] = acc_test

                    print('Evaluate %d random %s, mean = %.4f std = %.4f strategy = %s \n-------------------------'%(accs.shape[0], model_eval, np.mean(accs), np.std(accs), feature_strategy))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval][exp] =  accs

                ''' visualize and save '''
                # add pooling function name if it is not none
                save_name = os.path.join(args.save_path, f'vis_{model_args}_exp{exp}_iter{it}.png')
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

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
                        feat_syn = feature_syn[feat_idx, c*args.ipc:(c+1)*args.ipc].reshape((args.ipc,  feat_size[0], feat_size[1], feat_size[2]))    
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
                        # grad of CE loss of img_syn and grad of MSE loss of feat_syn
                        output_syn = net(img_syn)
                        loss_syn = criterion(output_syn, lab_syn) + args.lbd * feature_criterion(feature, feat_syn)
                        if isinstance(net, torch.nn.DataParallel):
                            output_feat_syn = net.module.get_output_from_features(feat_syn, layer_idx=args.layer_idx)
                        else:
                            output_feat_syn = net.get_output_from_features(feat_syn, layer_idx=args.layer_idx)
                        loss_feat_cls = criterion(output_feat_syn, lab_syn)
                        if it % 50 == 0:
                            with torch.no_grad():
                                acc_feat += torch.sum(torch.argmax(output_feat_syn.detach().clone(), dim=1) == lab_syn).item()
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters_wo_classifier, create_graph=True)
                        loss += match_loss(gw_syn, gw_real, args)
                        loss += loss_feat_cls * args.feat_lbd
                    acc_avg_feat[feat_idx] += acc_feat
                    optimizer_feat.zero_grad()
                    loss.backward()
                    optimizer_feat.step()
                    loss_avg_feature_matching[feat_idx] += loss.item()
                    loss_avg_feature_cls[feat_idx] += loss_feat_cls.item()
                    if ol == args.outer_loop - 1:
                        break

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                feat_syn_train = copy.deepcopy(feature_syn.detach()) # TODO: modified here
                dst_syn_train = TensorDatasetFeature(image_syn_train, label_syn_train, feat_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

                for il in range(args.inner_loop):
                    epoch_with_features('train', trainloader, net, optimizer_net, criterion, feature_criterion, args, aug = True if args.dsa else False, 
                                        lbd=args.lbd,
                                        pooling_function=pooling_function,
                                        layer_idx=args.layer_idx, 
                                        feat_loss=loss_avg_feature_cls, 
                                        feature_strategy=feature_strategy)

            loss_avg /= (num_classes*args.outer_loop)
            loss_avg_feature_matching /= (num_classes*args.outer_loop)
            loss_avg_feature_cls /= (num_classes*args.outer_loop)
            acc_avg_feat /= (num_classes*args.outer_loop*args.ipc)
            if it % 50 == 0:
                print('%s iter = %04d, Matching loss = %.4f' % (get_time(), it, loss_avg))
                print('feat matching loss = ', loss_avg_feature_matching)
                print('feat cls loss = ', loss_avg_feature_cls)
                print('feat cls acc = ', acc_avg_feat)
            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu()), copy.deepcopy(feature_syn.detach().cpu())])
                model_path =  os.path.join(args.save_path, f'model_{model_args}.pt')
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, model_path)
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key] # accs: (num_exp, num_eval)
        accs = accs.reshape(-1)
        print('Run %d experiments with Strategy %s, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, feature_strategy,args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


