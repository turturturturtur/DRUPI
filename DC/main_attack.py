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
from tqdm import tqdm
import torch.optim as optim

def project_noise(noise, epsilon):
    """
    Projects the noise tensor to have a norm less than or equal to epsilon.

    Args:
        noise: The noise tensor.
        epsilon: The maximum norm allowed.

    Returns:
        The projected noise tensor.
    """
    norm = noise.norm(p=2, dim=(1, 2, 3), keepdim=True)
    norm = torch.clamp(norm, min=epsilon)
    return noise / norm * epsilon


def optimize_images_n(
    net,
    images_train,
    labels_train,
    train_criterion,
    device,
    # pipe,
    num_iterations=1000,
    learning_rate=0.01,
    lbd=0.1,
    lbd_diff=0.1,
    bounded_noise=False,
    epsilon=1 / 255,
):
    """
    Optimize images by adding noise to minimize loss.

    Args:
        net: The neural network model.
        images_train: Training images.
        labels_train: Training labels.
        train_criterion: Training criterion.
        device: Device to run the optimization on.
        pipe: Diffusion model pipeline.
        num_iterations: Number of optimization iterations.
        learning_rate: Learning rate for the optimizer.
        lbd: Coefficient for score distillation loss.
        lbd_diff: Coefficient for difficulty loss.
        bounded_noise: Whether to bound the noise.
        epsilon: Epsilon value for bounded noise.

    Returns:
        The images with optimized noise.
    """
    # Set the model to evaluation mode

    net.eval()
    net_list=[]
    net_list.append(net)
    
    noise_list = []

    kk = 0

    for net in net_list:
        kk += 1
        if kk != 1:
            for i in tqdm(range(num_iterations), desc='num_iterations'):

                # Create a noise tensor and set it to require gradients
                if args.noise_init == 'zero':
                    noise = torch.zeros_like(
                        images_train, requires_grad=True, device=device)
                elif args.noise_init == 'one':
                    noise = torch.ones_like(
                        images_train, requires_grad=True, device=device)
                elif args.noise_init == 'rand':
                    noise = torch.rand_like(
                        images_train, requires_grad=True, device=device)

                # Load diffusion model to the device
                # pipe.to(device)

                # Define the optimizer to optimize the noise
                optimizer = optim.Adam([noise], lr=learning_rate)
                optimizer.zero_grad()
                # Add the noise to the images
                noisy_images = images_train + noise
                outputs = net(noisy_images)
                # Compute score distillation loss
                # all_same_logits = torch.ones_like(outputs) / outputs.shape[1]
                # loss_sd = get_score_distillation_loss(
                #     pipe, noisy_images, steps=args.step)
                # Compute difficulty loss
                # loss_diff = F.kl_div(F.log_softmax(outputs, dim=1), all_same_logits)
                # Total loss
                loss = (
                    train_criterion(outputs, labels_train)  # CrossEntropy
                    # + lbd * loss_sd
                    # + lbd_diff * loss_diff
                )
                # Backward pass
                loss.backward()
                # Update the noise using the optimizer
                optimizer.step()

                # Project the noise to the L2 norm ball if required
                if bounded_noise:
                    noise.data = project_noise(noise.data, epsilon)

                # Reduce learning rate after half iterations
                if i == num_iterations // 2:
                    for g in optimizer.param_groups:
                        g["lr"] = learning_rate / 10
                # Print loss every 10% of iterations
                # if i % (num_iterations // 10) == 0:
                #     print(f"Iteration {i}, Loss: {loss.item()}")
            noise_list.append(noise)

        # Return the images with optimized noise
        optimized_images = images_train + sum(noise_list)
        return optimized_images
    
def optimize_images(
    net,
    images_train,
    labels_train,
    train_criterion,
    device,
    num_iterations=100,
    learning_rate=0.01,
):

    # Set the model to evaluation mode

    net.eval()
    net=net.to(device)
    for i in tqdm(range(num_iterations), desc='num_iterations'):


        images_train = images_train.to(device)
        images_train.requires_grad_()
        optimizer = torch.optim.Adam([images_train], lr=learning_rate)


        optimizer.zero_grad()
        labels_train=labels_train.to(device)
        outputs = net(images_train)
        loss =  train_criterion(outputs, labels_train)  # CrossEntropy


        # Backward pass
        loss.backward()
        # Update the noise using the optimizer
        optimizer.step()

        # Reduce learning rate after half iterations
        if i == num_iterations // 2:
            for g in optimizer.param_groups:
                g["lr"] = learning_rate / 10


    optimized_images = images_train 

    return optimized_images 
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
    parser.add_argument('--lbd', type=float, default=0.1, help='scale of MSE')
    parser.add_argument('--lr_feat', type=float, default=0.1, help='learning rate for updating synthetic features')
    parser.add_argument('--layer-idx', type=int, default=None, help='layer index for feature')
    parser.add_argument('--pooling', type=str, default=None, help='feature pooling method')
    parser.add_argument('--feat-lbd', type=float, default=0.1, help='scale of CE')
    parser.add_argument('--feat-opt', type=str, default='SGD', help='featureoptimizer')
    parser.add_argument('--n-feat', type=int, default=1, help='number of features')
    parser.add_argument('--eval-freq', type=int, default=4, help='evaluate frequency')
    parser.add_argument('--use-feature', type=str, default='mean', help='the way to use the feature')
    parser.add_argument('--bounded_noise', action='store_true', default=False, help='normalize the feature')
    parser.add_argument('--feat-norm', action='store_true', default=False, help='normalize the feature')
    parser.add_argument('--feat_flag', action='store_true', default=False, help='the source of using feature_syn in the function,default is False means from dataloader,true means from pretrained_model')
    parser.add_argument('--feat-metric', type=str, default='MSE', help='feature criterion')
    parser.add_argument('--img-method',default='DC',help='image distillation method')
    parser.add_argument('--img-path', default='../distilled_data', help='image path')
    parser.add_argument('--iter_adv', type=int, default=100, help='attack iter') 
    parser.add_argument('--lr_adv', type=float, default=0.01, help='attack lr')    
    parser.add_argument('--add_noise', action='store_true', default=False, help='noise')    
    
    args = parser.parse_args()

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False
    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None
    data_path = os.path.expanduser(args.data_path)
    print(data_path)
    args.save_path = os.path.join(args.save_path, args.dataset, f'IPC{args.ipc}')
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    model_args = f'{args.dataset}_{args.model}_{args.ipc}ipc_iter-adv{args.iter_adv}__lr-adv{args.lr_adv}__[MSE_{args.lbd}_pool{args.pooling}_nfeat{args.n_feat}_use{args.use_feature}_norm{args.feat_norm}]_{args.img_method}_noise{args.add_noise}__{get_time()}'
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
    eval_it_pool = np.arange(0, args.Iteration+1, args.Iteration//args.eval_freq).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    if args.feat_metric == 'MSE':
        feature_criterion = nn.MSELoss()
    elif args.feat_metric == 'MMD':
        feature_criterion = MMDLoss()
    elif args.feat_metric == 'cosine':
        feature_criterion = CosineLoss()
    # if args.contrast:
    #     adversarial_criterion = SupConLoss()
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
        
        # ===================================
        print('%s training begins'%get_time())

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

        pt_file_path = '/data3/yyt/FD/DC/result_true/CIFAR10/IPC50/model_CIFAR10_ConvNet_50ipc_[MSE_0.005_poolNone_layerNone_CE0.01_nfeat1_usemean_normFalse]_DC_[2024-08-21 08:47:12]_poolingFlagFalse_imgUdpTrue.pt'  
        checkpoint = torch.load(pt_file_path)
        data_save=checkpoint['data']
        print("=================before attack====================")
        print(np.mean(checkpoint['accs_all_exps']['ConvNet']))
        print("=================before attack====================")


        for idx, (image_syn, label_syn, feature_syn) in enumerate(data_save):
            print(f" image_syn {idx+1} shape : {image_syn.shape}")
            print(f" label_syn {idx+1} shape : {label_syn.shape}")
            print(f" feature_syn {idx+1} shape : {feature_syn.shape}")
        if args.add_noise==True:
            img=optimize_images_n(pretrained_net, image_syn,label_syn,train_criterion=criterion,device=args.device,num_iterations=args.iter_adv,learning_rate=args.lr_adv)
        # img=optimize_images(pretrained_net, image_syn,label_syn,train_criterion=criterion,device=args.device,num_iterations=args.iter_adv,learning_rate=args.lr_adv)
        else:
            img=optimize_images(pretrained_net, image_syn,label_syn,train_criterion=criterion,device=args.device,num_iterations=args.iter_adv,learning_rate=args.lr_adv)
        img=img.float().to(args.device)
        feat_loss = None
        accs = np.zeros((args.num_eval))
        model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

        for model_eval in model_eval_pool:
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
            print('DC augmentation parameters: \n', args.dc_aug_param)
            for it_eval in range(args.num_eval):
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                image_syn_eval, label_syn_eval = copy.deepcopy(img.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                feature_syn_eval = copy.deepcopy(feature_syn.detach())

                _, acc_train, acc_test = evaluate_synset_w_feature(it_eval, net_eval, image_syn_eval, label_syn_eval, feature_syn_eval, testloader, args, feature_criterion, pooling_function=pooling_function,feat_loss=feat_loss,feature_strategy=feature_strategy)
                accs[it_eval] = acc_test
        print('Evaluate %d random %s, mean = %.4f std = %.4f strategy = %s \n-------------------------'%(accs.shape[0], model_eval, np.mean(accs), np.std(accs), feature_strategy))
        # ===================================
        
if __name__ == '__main__':
    main()
