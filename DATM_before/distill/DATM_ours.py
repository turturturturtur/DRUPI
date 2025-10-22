import os
import sys
sys.path.append("../")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug,TensorDataset,epoch, epoch_with_features, evaluate_synset_w_feature
import wandb
import copy
import random
from reparam_module import ReparamModule
# from kmeans_pytorch import kmeans
from utils.cfg import CFG as cfg
import warnings
import yaml

#
from functools import partial
import matplotlib.pyplot as plt
from feature_metric import MMDLoss, CosineLoss, SupConLoss
import os.path as osp 
# 

warnings.filterwarnings("ignore", category=DeprecationWarning)

def manual_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def main(args):

    manual_seed()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.device])

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 
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

    if args.feat_metric == 'MSE':
        feature_criterion = nn.MSELoss()
    elif args.feat_metric == 'MMD':
        feature_criterion = MMDLoss()
    elif args.feat_metric == 'cosine':
        feature_criterion = CosineLoss()
    if args.contrast:
        adversarial_criterion = SupConLoss()

# 

    if args.skip_first_eva==False:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(sync_tensorboard=False,
                project="DATM_tesla",
                job_type="CleanRepo",
                config=args,
                name="ALIYUN",
                )
    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    if args.dataset == 'ImageNet1K' and os.path.exists('images_all.pt') and os.path.exists('labels_all.pt'):
        images_all = torch.load('images_all.pt')
        labels_all = torch.load('labels_all.pt')
    else:
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
        images_all = torch.cat(images_all, dim=0).to("cpu")
        labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
        if args.dataset == 'ImageNet1K':
            torch.save(images_all, 'images_all.pt')
            torch.save(labels_all, 'labels_all.pt')

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)


    # 
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


    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]



    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))
    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        # random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]

        expert_id = [i for i in range(len(expert_files))]
        random.shuffle(expert_id)

        print("loading file {}".format(expert_files[expert_id[file_idx]]))
        buffer = torch.load(expert_files[expert_id[file_idx]])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        buffer_id = [i for i in range(len(buffer))]
        random.shuffle(buffer_id)

    if args.pix_init == 'real':
        print('initialize synthetic data from pretrained data')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
            
    
    elif args.pix_init == 'samples_predicted_correctly':
        if args.parall_eva==False:
            device = torch.device("cuda:0")
        else:
            device = args.device
        # if cfg.Initialize_Label_With_Another_Model:
        #     Temp_net = get_network(args.Initialize_Label_Model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        # else:
        #     Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        # Temp_net.eval()
        # Temp_net = ReparamModule(Temp_net)
        # if args.distributed and args.parall_eva==True:
        #     Temp_net = torch.nn.DataParallel(Temp_net)
        # Temp_net.eval()
        # logits=[]
        label_expert_files = expert_files
        # temp_params = torch.load(label_expert_files[0])[0][args.Label_Model_Timestamp]
        # temp_params = torch.cat([p.data.to(device).reshape(-1) for p in temp_params], 0)
        # if args.distributed and args.parall_eva==True:
        #     temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        # for c in range(num_classes):
        #     data_for_class_c = get_images(c, len(indices_class[c])).detach().data
        #     n, _, w, h = data_for_class_c.shape
        #     selected_num = 0
        #     select_times = 0
        #     cur=0
        #     temp_img = None
        #     Wrong_Predicted_Img = None
        #     batch_size = 256
        #     index = []
        #     while len(index)<args.ipc:
        #         print(str(c)+'.'+str(select_times)+'.'+str(cur))
        #         current_data_batch = data_for_class_c[batch_size*select_times : batch_size*(select_times+1)].detach().to(device)
        #         if batch_size*select_times > len(data_for_class_c):
        #             select_times = 0
        #             cur+=1
        #             temp_params = torch.load(label_expert_files[int(cur/10)%10])[cur%10][args.Label_Model_Timestamp]
        #             temp_params = torch.cat([p.data.to(device).reshape(-1) for p in temp_params], 0).to(device)
        #             if args.distributed and args.parall_eva==True:
        #                 temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        #             continue
        #         logits = Temp_net(current_data_batch, flat_param=temp_params).detach()
        #         prediction_class = np.argmax(logits.cpu().data.numpy(), axis=-1)
        #         for i in range(len(prediction_class)):
        #             if prediction_class[i]==c and len(index)<args.ipc:
        #                 index.append(batch_size*select_times+i)
        #                 index=list(set(index))
        #         select_times+=1
        #         if len(index) == args.ipc:
        #             temp_img = torch.index_select(data_for_class_c, dim=0, index=torch.tensor(index))
        #             break
        #     image_syn.data[c * args.ipc:(c + 1) * args.ipc] = temp_img.detach()
        image_syn_pretrained = torch.load(osp.join(load_path, f'images_best.pt')).to(args.device)
        image_syn = image_syn_pretrained.clone().detach().requires_grad_(True).to(args.device)
        feature_syn = []
        for _ in range(args.n_feat):
            pretrained_net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            pretrained_net.train()
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
            image_train, label_train = copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_train, label_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
            optimizer_pretrained = torch.optim.SGD(pretrained_net.parameters(), lr=0.01) 
            optimizer_pretrained.zero_grad()
            criterion = nn.CrossEntropyLoss().to(args.device)
            epoch('train', trainloader, pretrained_net, optimizer_pretrained, criterion, args, False)
            if isinstance(pretrained_net, torch.nn.DataParallel):
                feature_syn_single = pretrained_net.module.get_features(image_syn_train, args.layer_idx).detach().clone()
            else:
                feature_syn_single = pretrained_net.get_features(image_syn_train, args.layer_idx).detach().clone()
            # copy feature_syn into n_feat in the first dimension
            feature_syn.append(feature_syn_single)
            # del pretrained_net
        feature_syn = torch.stack(feature_syn, dim=0).to(args.device)
    else:
        print('initialize synthetic data from random noise')
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        feature_syn = torch.randn(size=(args.n_feat, num_classes*args.ipc, feat_size[0], feat_size[1], feat_size[2]), dtype=torch.float, requires_grad=True, device=args.device)


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    # 
    # add synthetic features optimizer
    if args.feat_opt == 'SGD':
        optimizer_feat = torch.optim.SGD([feature_syn, ], lr=args.lr_feat, momentum=0.5) # optimizer_feat for synthetic features
    else:
        optimizer_feat = torch.optim.Adam([feature_syn, ], lr=args.lr_feat)
    optimizer_feat.zero_grad()
    feature_criterion = feature_criterion.to(args.device)

    # 
    
    optimizer_img.zero_grad()

    ###

    '''test'''
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    criterion = SoftCrossEntropy

    print('%s training begins'%get_time())
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    '''------test------'''
    '''only sum correct predicted logits'''
    if args.pix_init == "samples_predicted_correctly":
        if cfg.Initialize_Label_With_Another_Model:
            Temp_net = get_network(args.Initialize_Label_Model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        else:
            Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        Temp_net.eval()
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        logits=[]
        batch_size = 256
        for i in range(len(label_expert_files)):
            Temp_Buffer = torch.load(label_expert_files[i])
            for j in Temp_Buffer:
                temp_logits = None
                for select_times in range((len(image_syn)+batch_size-1)//batch_size):
                    current_data_batch = image_syn[batch_size*select_times : batch_size*(select_times+1)].detach().to(device)
                    Temp_params = j[args.Label_Model_Timestamp]
                    Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
                    if args.distributed:
                        Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                    Initialized_Labels = Temp_net(current_data_batch, flat_param=Initialize_Labels_params)
                    if temp_logits == None:
                        temp_logits = Initialized_Labels.detach()
                    else:
                        temp_logits = torch.cat((temp_logits, Initialized_Labels.detach()),0)
                logits.append(temp_logits.detach().cpu())
        logits_tensor = torch.stack(logits)
        true_labels = label_syn.cpu()
        predicted_labels = torch.argmax(logits_tensor, dim=2).cpu()
        correct_predictions = predicted_labels == true_labels.view(1, -1)
        mask = correct_predictions.unsqueeze(2)
        correct_logits = logits_tensor * mask.float()
        correct_logits_per_model = correct_logits.sum(dim=0)
        num_correct_images_per_model = correct_predictions.sum(dim=0, dtype=torch.float)
        average_logits_per_image = correct_logits_per_model / num_correct_images_per_model.unsqueeze(1) 
        Initialized_Labels = average_logits_per_image

    elif args.pix_init == "real":
        Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        Temp_params = buffer[0][-1]
        Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
        if args.distributed:
            Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        Initialized_Labels = Temp_net(image_syn, flat_param=Initialize_Labels_params)

    acc = np.sum(np.equal(np.argmax(Initialized_Labels.cpu().data.numpy(), axis=-1), label_syn.cpu().data.numpy()))
    print('InitialAcc:{}'.format(acc/len(label_syn)))

    label_syn = copy.deepcopy(Initialized_Labels.detach()).to(args.device).requires_grad_(True)
    label_syn.requires_grad=True
    label_syn = label_syn.to(args.device)
    

    optimizer_y = torch.optim.SGD([label_syn], lr=args.lr_y, momentum=args.Momentum_y)
    vs = torch.zeros_like(label_syn)
    accumulated_grad = torch.zeros_like(label_syn)
    last_random = 0

    del Temp_net

    # test
    curMax_times = 0
    current_accumulated_step = 0

    for it in range(0, args.Iteration+1):
        save_this_it = False
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []

                
# 
                if it == 0:
                    feat_loss = None
                else:
                    feat_loss = loss_avg_feature_cls
# 
                for it_eval in range(args.num_eval):
                    if args.parall_eva==False:
                        device = torch.device("cuda:0")
                        net_eval = get_network(model_eval, channel, num_classes, im_size, dist=False).to(device) # get a random model
                    else:
                        device = args.device
                        net_eval = get_network(model_eval, channel, num_classes, im_size, dist=True).to(device) # get a random model

                    eval_labs = label_syn.detach().to(device)
                    with torch.no_grad():
                        image_save = image_syn.to(device)
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()).to(device), copy.deepcopy(eval_labs.detach()).to(device) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    # 
                    feature_syn_eval = copy.deepcopy(feature_syn.detach())
                    if it_eval == 0:
                        print('Evaluate without feature!')
                        _, acc_train, acc_test = evaluate_synset(it_eval, copy.deepcopy(net_eval).to(device), image_syn_eval.to(device), label_syn_eval.to(device), testloader, args, texture=False, train_criterion=criterion)
                    else:
                        _, acc_train, acc_test = evaluate_synset_w_feature(it_eval, copy.deepcopy(net_eval).to(device), image_syn_eval.to(device), label_syn_eval.to(device), feature_syn_eval, testloader, args,feature_criterion,  texture=False, train_criterion=criterion,pooling_function=pooling_function,feat_loss=feat_loss,feature_strategy=feature_strategy)
#   
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)

                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)

                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()
                save_dir = os.path.join(".", "logged_files", args.dataset, str(args.ipc), args.model, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(os.path.join(save_dir,'Normal'))
                    
                torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal',"images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_{}.pt".format(it)))
                torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal', "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_best.pt".format(it)))
                    torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()
                        torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal', "images_zca_{}.pt".format(it)))
                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)
                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)


     
        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()
        
# 
        loss_avg_feature_matching = np.zeros(args.n_feat)
        loss_avg_feature_cls = np.zeros(args.n_feat)
        acc_avg_feat = np.zeros(args.n_feat)
# 

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[buffer_id[expert_idx]]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_id)
                print("loading file {}".format(expert_files[expert_id[file_idx]]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[expert_id[file_idx]])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer_id)

        # Only match easy traj. in the early stage
        if args.Sequential_Generation:
            Upper_Bound = args.current_max_start_epoch + int((args.max_start_epoch-args.current_max_start_epoch) * it/(args.expansion_end_epoch))
            Upper_Bound = min(Upper_Bound, args.max_start_epoch)
        else:
            Upper_Bound = args.max_start_epoch

        start_epoch = np.random.randint(args.min_start_epoch, Upper_Bound)

        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn
        y_hat = label_syn
        syn_feat = feature_syn

        param_loss_list = []
        param_dist_list = []
        indices_chunks  = []  

        
        acc_feat_total = 0
        loss_feat_step_total = torch.tensor(0.0).to(args.device)

        for step in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            this_feat = syn_feat[these_indices]

            print(this_feat.shape)
            
            optimizer_feat.zero_grad()
            # print('before',x.shape)
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            # print('after',x.shape)
            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]


            criterion_feature = feature_criterion.to(args.device)
            if feature_strategy is None:
                feature_strategy = "mean"
            if loss_avg_feature_cls is None:
                loss_avg_feature_cls = np.zeros(args.n_feat)
            this_feat = this_feat.float().to(args.device)
            if feature_strategy == "mean":
                this_feat = torch.mean(this_feat, dim=1)  # use average feature
            elif feature_strategy == "random":
                this_feat = this_feat[
                    :, torch.randint(0, this_feat.shape[1], (1,)).item()
                ]
            elif feature_strategy == "max":
                best_feature = np.argmin(feat_loss)
                this_feat = this_feat[:, best_feature]
            layer_idx = args.layer_idx
            lbd = args.lbd
            if isinstance(student_net, torch.nn.DataParallel):
                feature = student_net.module.get_features(x, layer_idx)
            else:
                feature = student_net.get_features(x, layer_idx)
            if pooling_function:
                feature = pooling_function(feature)
                this_feat = pooling_function(this_feat)
            if args.feat_norm:
                feature = torch.nn.functional.normalize(feature, p=2, dim=1)
                this_feat = torch.nn.functional.normalize(this_feat, p=2, dim=1)

            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            loss_ce = ce_loss
            loss_reg = criterion_feature(feature, this_feat)
            loss_syn = loss_ce + lbd * loss_reg

            grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)

            if isinstance(student_net, torch.nn.DataParallel):
                output_feat_syn = student_net.module.get_output_from_features(this_feat, layer_idx=args.layer_idx)
                output_feat_syn = student_net.module.classifier(this_feat.view(this_feat.size(0), -1))
            else:
                output_feat_syn = student_net.get_output_from_features(this_feat, layer_idx=args.layer_idx)
                output_feat_syn = student_net.classifier(this_feat.view(this_feat.size(0), -1))

            loss_feat_step = criterion(output_feat_syn, this_y)
            loss_feat_step_total += loss_feat_step
            with torch.no_grad():
                print(output_feat_syn.detach().clone())
                print(torch.argmax(output_feat_syn.detach().clone(), dim=1).shape)
                print(this_y)
                acc_feat_total += torch.sum(torch.argmax(output_feat_syn.detach().clone(), dim=1) == this_y.detach()).item()

        

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)
        
        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        loss_feat_total = grand_loss + loss_feat_step_total * args.feat_lbd
        # 需要一个不用类别的supconfloss
        # if args.contrast:
        #     '''add contrastive SupConf loss'''
        #     loss_adv = compute_info_nce_loss(feature_syn[feat_idx].clone(), label_syn, c)
        #     loss += loss_adv * args.lbd_contrast
        loss_feat_total.backward()

        loss_avg_feature_matching = loss_feat_total / args.syn_steps
        loss_avg_feature_step = loss_feat_step_total / args.syn_steps
        acc_avg_feat = acc_feat_total / args.syn_steps
        acc_avg_feat = f'{acc_avg_feat*100}%'
        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        optimizer_y.zero_grad()
        optimizer_feat.zero_grad()
        
        
        grand_loss.backward()

        if grand_loss<=args.threshold:
            optimizer_y.step()
            optimizer_img.step()
            optimizer_lr.step()
            optimizer_feat.step()
        else:
            wandb.log({"falts": start_epoch}, step=it)



        wandb.log({"Grand_Loss": param_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
        

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument("--cfg", type=str, default="")
    # 
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
    parser.add_argument('--lbd-contrast', type=float, default=0.1, help='scale of contrastive loss')
    parser.add_argument('--feat-metric', type=str, default='MSE', help='feature criterion')
    parser.add_argument('--img-method',default='DATM',help='image distillation method')
    parser.add_argument('--img-path', default='../../distilled_data', help='image path')
# 
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()
    main(args)



