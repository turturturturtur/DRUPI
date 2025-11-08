import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug,FlushFile,evaluate_synset_w_feature,TensorDataset,epoch
# import wandb
import copy
import random
from functools import partial
from reparam_module import ReparamModule
from feature_metric import MMDLoss, CosineLoss, SupConLoss
import os.path as osp
import warnings
from torchvision.utils import save_image
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 


def manual_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def safe_avg_pool2d(x, kernel_size=2, stride=2, padding=0, ceil_mode=False):
    if x.dim() == 4:
        h, w = x.shape[-2:]
        if h < kernel_size or w < kernel_size:
            # 太小就不池化，直接返回
            return x
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride,
                        padding=padding, ceil_mode=ceil_mode)
def main(args):
    # manual_seed()
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    #
    if args.dataset == 'ImageNet':
        load_path = osp.join(args.img_path, args.img_method, args.dataset,args.subset, f"IPC{args.ipc}")
    else:
        load_path = osp.join(args.img_path, args.img_method, args.dataset, f"IPC{args.ipc}")
    feature_strategy = args.use_feature
    print("feature_strategy: ", feature_strategy)
    feat_size = [128, 4, 4]
    if args.layer_idx:
        if args.layer_idx == 1:
            feat_size = [128, 16, 16]
        elif args.layer_idx == 2:
            feat_size = [128, 8, 8]
    if args.pooling:
        feat_size = [128, 1, 1]

    if args.feat_metric == "MSE":
        feature_criterion = nn.MSELoss()
    elif args.feat_metric == "MMD":
        feature_criterion = MMDLoss()
    elif args.feat_metric == "cosine":
        feature_criterion = CosineLoss()
    if args.contrast:
        adversarial_criterion = SupConLoss()

    #

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
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

    # wandb.init(sync_tensorboard=False,
    #            project="DatasetDistillation",
    #            job_type="CleanRepo",
    #            config=args,
    #            )

    # args = type('', (), {})()

    # for key in wandb.config._items:
    #     setattr(args, key, wandb.config._items[key])

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
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[int(sample[1])])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
        
    # define pooling function
    pooling_function = None
    if args.pooling:
        if args.pooling == "avg":
            # avgpooling2d function
            pooling_function = partial(
                F.adaptive_avg_pool2d, output_size=(1, 1)
            )
        elif args.pooling == "max":
            # maxpooling2d function
            pooling_function = partial(
                F.adaptive_max_pool2d, output_size=(1, 1)
            )
        elif args.pooling == "sum":
            # sumpooling2d function
            pooling_function = partial(
                lambda t: t.sum(dim=(-2, -1), keepdim=True)
            )
        else:
            raise NotImplementedError("Pooling method not implemented")
    #
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    # ==== (NEW) generate_pretrained: 从真实数据抽样并生成 images_best.pt ====
    if args.generate_pretrained:
        # 每类随机抽 args.ipc 张，按 [num_classes*ipc, C, H, W] 组织
        image_syn = torch.empty(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
                                dtype=torch.float32, device='cpu')
        for c in range(num_classes):
            idx_shuffle = np.random.permutation(indices_class[c])[:args.ipc]
            image_syn[c*args.ipc:(c+1)*args.ipc] = images_all[idx_shuffle]  # images_all 在前面已在 CPU

        os.makedirs(load_path, exist_ok=True)
        save_file = osp.join(load_path, "images_best.pt")
        torch.save(image_syn.detach().cpu(), save_file)
        print(f"[OK] Pretrained images saved to: {save_file}")
        print("You can now run training with --pix_init real (without --generate_pretrained).")
        return  # 只做预生成则直接退出



    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]


    ''' initialize the synthetic data '''
    label_syn = torch.arange(num_classes, device=args.device, dtype=torch.long).repeat_interleave(args.ipc) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        pretrained_file = osp.join(load_path, "images_best.pt")
        if not osp.exists(pretrained_file):
            raise FileNotFoundError(
                f"Missing pretrained file: {pretrained_file}\n"
                f"请先运行带 --generate_pretrained 的命令预生成该文件，再用 --pix_init real 训练。"
            )
        image_syn_pretrained = torch.load(pretrained_file).to(args.device)
        image_syn = image_syn_pretrained.clone().detach().requires_grad_(True).to(args.device)
        del image_syn_pretrained
        feature_syn = []
        for _ in range(args.n_feat):
            pretrained_net = get_network(args.model, channel, num_classes, im_size).to(
                args.device
            )  # get a random model
            pretrained_net.train()
            image_syn_train, label_syn_train = copy.deepcopy(
                image_syn.detach()
            ), copy.deepcopy(label_syn.detach())
            image_train, label_train = copy.deepcopy(
                images_all.detach()
            ), copy.deepcopy(
                labels_all.detach()
            )  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_train, label_train)
            trainloader = torch.utils.data.DataLoader(
                dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0
            )
            optimizer_pretrained = torch.optim.SGD(pretrained_net.parameters(), lr=0.01)
            optimizer_pretrained.zero_grad()
            criterion = nn.CrossEntropyLoss().to(args.device)
            epoch(
                "train",
                trainloader,
                pretrained_net,
                optimizer_pretrained,
                criterion,
                args,
                aug=False
            )
            if isinstance(pretrained_net, torch.nn.DataParallel):
                feature_syn_single = (
                    pretrained_net.module.get_features(image_syn_train, args.layer_idx)
                    .detach()
                    .clone()
                )
            else:
                feature_syn_single = (
                    pretrained_net.get_features(image_syn_train, args.layer_idx)
                    .detach()
                    .clone()
                )
            # copy feature_syn into n_feat in the first dimension
            feature_syn.append(feature_syn_single)
            del pretrained_net
        feature_syn = torch.stack(feature_syn, dim=0).to(args.device)
        feature_syn.requires_grad_()
    else:
        print('initialize synthetic data from random noise')

    # print('img_syn,lab_syn,feature_syn',image_syn.shape,feature_syn.shape)

    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    #
    # add synthetic features optimizer
    if args.feat_opt == "SGD":
        optimizer_feat = torch.optim.SGD(
            [
                feature_syn,
            ],
            lr=args.lr_feat,
            momentum=0.5,
        )  
    else:
        optimizer_feat = torch.optim.Adam(
            [
                feature_syn,
            ],
            lr=args.lr_feat,
        )
    optimizer_feat.zero_grad()
    feature_criterion = feature_criterion.to(args.device)

    #

    criterion = nn.CrossEntropyLoss().to(args.device)

    # ========= Checkpoint: 收集已创建对象 / 定义保存与加载 / 可选恢复 =========

    # 统一的 ckpt 目录，和你保存图片的目录保持同一 run 名称（model_args）
    ckpt_dir  = osp.join(".", "logged_files", args.dataset, model_args, "ckpts")
    ckpt_last = osp.join(ckpt_dir, "ckpt_last.pt")
    ckpt_best = osp.join(ckpt_dir, "ckpt_best.pt")

    # 收集你已经创建好的优化器（名字跟你的代码一致）
    optimizers = {}
    if 'optimizer_img' in locals() and optimizer_img is not None:
        optimizers['img']  = optimizer_img
    if 'optimizer_lr' in locals() and optimizer_lr is not None:
        optimizers['lr']   = optimizer_lr
    if 'optimizer_feat' in locals() and optimizer_feat is not None:
        optimizers['feat'] = optimizer_feat

    # 你的脚本目前没用 scheduler & 混合精度，先占位（有就会加载，没有就忽略）
    schedulers = {}
    scaler = locals().get('scaler', None)

    def _save_ckpt(path, it, best_acc_dict):
        state = {
            # 关键：把可训练“张量参数”都存起来（注意不是 student_net；student_net 每个 it 都重建）
            'image_syn': image_syn.detach().cpu(),
            'feature_syn': feature_syn.detach().cpu(),
            'syn_lr': syn_lr.detach().cpu(),
            # 优化器（官方建议同时保存）：
            'optimizers': {k: opt.state_dict() for k, opt in optimizers.items()},
            'schedulers': {k: sch.state_dict() for k, sch in schedulers.items()},
            'scaler': scaler.state_dict() if scaler is not None else None,
            # 迭代与指标
            'iter': int(it),
            'best_acc_dict': best_acc_dict,
            'args': vars(args),
        }
        os.makedirs(osp.dirname(path), exist_ok=True)
        torch.save(state, path)

    def _load_ckpt(path):
        ckpt = torch.load(path, map_location=args.device)
        # 1) 把保存的“张量参数”拷回现有张量（保持 id 不变，优化器 state 可无缝对上）
        image_syn.data.copy_(ckpt['image_syn'].to(args.device))
        feature_syn.data.copy_(ckpt['feature_syn'].to(args.device))
        syn_lr.data.copy_(ckpt['syn_lr'].to(args.device))

        # 2) 恢复优化器/调度器
        for k, opt in optimizers.items():
            state = ckpt.get('optimizers', {}).get(k, None)
            if state is not None:
                opt.load_state_dict(state)
        for k, sch in schedulers.items():
            state = ckpt.get('schedulers', {}).get(k, None)
            if sch is not None and state is not None:
                sch.load_state_dict(state)

        # 3) AMP（若你后来加了 amp，建议也一起保存/加载 scaler）
        if scaler is not None and ckpt.get('scaler', None) is not None:
            scaler.load_state_dict(ckpt['scaler'])

        it  = int(ckpt.get('iter', 0))
        best_dict = ckpt.get('best_acc_dict', {})
        best_scalar = float(max(best_dict.values())) if len(best_dict) > 0 else -1.0
        print(f"[resume] loaded: {path} | iter={it} | best_acc={best_scalar:.2f}")
        return it, best_dict

    # 是否要从断点恢复
    start_it = 0
    best_acc_loaded = None
    if getattr(args, 'resume_from', '') and osp.isfile(args.resume_from):
        start_it, best_acc_loaded = _load_ckpt(args.resume_from)




    print('%s training begins'%get_time())

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
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}


    # === 进度条：从 start_it → args.Iteration（断点续训也能显示整体进度） ===
    pbar = tqdm(range(start_it, args.Iteration+1), initial=start_it, total=args.Iteration+1,
                dynamic_ncols=True, desc="Train")
    for it in pbar:
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        # wandb.log({"Progress": it}, step=it)
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
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    feature_syn_eval = copy.deepcopy(feature_syn.detach())
                    if it_eval == 0:
                        print("Evaluate without feature!")
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    else:
                        _, acc_train, acc_test = evaluate_synset_w_feature(it_eval, net_eval, image_syn_eval, label_syn_eval,feature_syn_eval, testloader, args,feature_criterion, texture=args.texture,pooling_function=pooling_function,
                            feat_loss=feat_loss,
                            feature_strategy=feature_strategy,)
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
                print('Evaluate %d random %s, mean = %.4f std = %.4f strategy = %s \n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std,feature_strategy))
                # wandb.log({"Accuracy/{}".format(model_eval): acc_test_mean}, step=it)
                print('iter = {},model_eval = {}, Accuracy = {},'.format(it,model_eval,acc_test_mean))
                # wandb.log(
                #     {"Max_Accuracy/{}".format(model_eval): best_acc[model_eval]},
                #     step=it,
                # )
                print('iter = {},model_eval = {}, Max_Accuracy = {},'.format(it,model_eval, best_acc[model_eval]))
                # wandb.log({"Std/{}".format(model_eval): acc_test_std}, step=it)
                print('iter = {},model_eval = {}, Std = {},'.format(it,model_eval,acc_test_std))
                # wandb.log(
                #     {"Max_Std/{}".format(model_eval): best_std[model_eval]}, step=it
                # )
                print('iter = {},model_eval = {}, Max_Std = {},'.format(it,model_eval,best_std[model_eval]))


        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()
                feature_save = feature_syn.cuda()

                # save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)
                save_dir = os.path.join(".", "logged_files", args.dataset, model_args)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))
                torch.save(feature_save.cpu(), os.path.join(save_dir, "features_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(feature_save.cpu(), os.path.join(save_dir, "features_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

                # wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    save_path = os.path.join(save_dir, 'Synthetic_Images')
                    if not os.path.exists(save_path):
                        os.makedirs(os.path.join(save_path))
                    image_data = torch.nan_to_num(grid.detach().cpu())

                    # 保存图像
                    # vutils.save_image(image_data, f"synthetic_image_step_{it}.png")
                    save_name = os.path.join(save_path, f'{model_args}_iter{it}.png')
                    save_image(image_data, save_name)
                    # wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    # wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        # wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        save_path = os.path.join(save_dir, f'Clipped_Synthetic_Images/std_{clip_val}')
                        if not os.path.exists(save_path):
                            os.makedirs(os.path.join(save_path))
                        # 将图像数据从 GPU 转移到 CPU，并去除 NaN 值
                        image_data = torch.nan_to_num(grid.detach().cpu())

                        # 保存图像
                        # vutils.save_image(image_data, f"synthetic_image_step_{it}.png")
                        save_name = os.path.join(save_path, f'{model_args}_iter{it}.png')
                        save_image(image_data, save_name, nrow=args.ipc)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        # wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        save_path = os.path.join(save_dir,'Reconstructed_Images')
                        if not os.path.exists(save_path):
                            os.makedirs(os.path.join(save_path))
                        # 将图像数据从 GPU 转移到 CPU，并去除 NaN 值
                        image_data = torch.nan_to_num(grid.detach().cpu())

                        # 保存图像
                        # vutils.save_image(image_data, f"synthetic_image_step_{it}.png")
                        save_name = os.path.join(save_path, f'{model_args}_iter{it}.png')
                        save_image(image_data, save_name, nrow=args.ipc)
                        # wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            # wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                # torch.nan_to_num(grid.detach().cpu()))}, step=it)
                            save_path = os.path.join(save_dir, f'Clipped_Reconstructed_Images/std_{clip_val}')
                            if not os.path.exists(save_path):
                                os.makedirs(os.path.join(save_path))
                            # 将图像数据从 GPU 转移到 CPU，并去除 NaN 值
                            image_data = torch.nan_to_num(grid.detach().cpu())

                            # 保存图像
                            # vutils.save_image(image_data, f"synthetic_image_step_{it}.png")
                            save_name = os.path.join(save_path, f'{model_args}_iter{it}.png')
                            save_image(image_data, save_name, nrow=args.ipc)
            # === 在这个大 if 的末尾追加：保存 ckpt ===
            # 1) 按固定频率存“最新”
            if (it + 1) % args.save_every == 0:
                _save_ckpt(ckpt_last, it + 1, best_acc)

            # 2) 如果这次评估刷新最好，也顺手存“最优”
            if save_this_it:
                _save_ckpt(ckpt_best, it + 1, best_acc)
        # wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()
        
        #
        loss_avg_feature_matching = np.zeros(args.n_feat)
        loss_avg_feature_cls = np.zeros(args.n_feat)
        acc_avg_feat = np.zeros(args.n_feat)

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn
        syn_feat = feature_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        acc_feat_total = 0
        loss_feat_step_total = torch.tensor(0.0).to(args.device)

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            this_y = y_hat[these_indices]
            if feature_strategy is None:
                feature_strategy = "mean"
            this_feat = syn_feat[:,these_indices]
            optimizer_feat.zero_grad()
            
            # if step ==0:
            #     print('x this_y this_feat',x.shape,this_y.shape,this_feat.shape)
            #     print('this_y',this_y)
            
            criterion_feature = feature_criterion.to(args.device)
            
            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
                
            criterion_feature = feature_criterion.to(args.device)    
                
            # update network
            this_feat = this_feat.float().to(args.device)
            if feature_strategy is None:
                feature_strategy = "mean"
            if loss_avg_feature_cls is None:
                loss_avg_feature_cls = np.zeros(args.n_feat)
            if feature_strategy == "mean":
                this_feat = torch.mean(this_feat, dim=0)  # use average feature
                
            elif feature_strategy == "random":
                this_feat = this_feat[
                torch.randint(0, this_feat.shape[0], (1,)).item()
                ]
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

            # if step ==0:
            #     print('aft mean  feature this_feat',feature.shape,this_feat.shape)
            avg_pool = nn.AvgPool2d(kernel_size=4)

            this_feat_update = this_feat

            # if step == 0:
            #     print('bef shape',feature.shape,this_feat.shape)
            feature = safe_avg_pool2d(feature, kernel_size=2, stride=2)
            this_feat = safe_avg_pool2d(this_feat, kernel_size=2, stride=2)
            # if step ==0:
            #     print('aft pool feature this_feat',feature.shape,this_feat.shape)

            if feature.size(1) != this_feat.size(1):           
                fc1=nn.Linear(feature.size(1), this_feat.size(1))
                feature = feature.permute(0, 2, 3, 1)
                feature=fc1(feature)
                feature = feature.permute(0, 3, 1, 2)
            
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)
            loss_ce = ce_loss
            loss_reg = criterion_feature(feature, this_feat)
            loss_syn = loss_ce + lbd * loss_reg

            grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)
            # update features
            if isinstance(student_net, torch.nn.DataParallel):
                # this_feat_fixed = this_feat_fixed.squeeze(0)
                output_feat_syn = student_net.module.get_output_from_features(
                    this_feat_update, layer_idx=args.layer_idx
                )
            else:
                # this_feat_fixed = this_feat_fixed.squeeze(0)
                output_feat_syn = student_net.get_output_from_features(
                    this_feat_update, layer_idx=args.layer_idx
                )
                
                
            # if step == 0:
            #     print('output_feat_syn this_y',output_feat_syn.shape,this_y.shape)
            #     print('output_feat_syn',output_feat_syn)
                # output_feat_syn this_y torch.Size([10, 10]) torch.Size([10])
#                 x this_y this_feat torch.Size([10, 3, 32, 32]) torch.Size([10]) torch.Size([3, 10, 128, 4, 4])
# aft mean  feature this_feat torch.Size([10, 128, 4, 4]) torch.Size([3, 128, 4, 4])
# aft pool feature this_feat torch.Size([10, 128, 1, 1]) torch.Size([3, 128, 1, 1])

#                 x, this_y this_feat torch.Size([100, 3, 32, 32]) torch.Size([100, 10]) torch.Size([1, 100, 128, 4, 4])
                
            loss_feat_step = criterion(output_feat_syn, this_y)
            loss_feat_step_total += loss_feat_step
            with torch.no_grad():
                acc_feat_total += torch.sum(
                    torch.argmax(output_feat_syn, dim=1) == torch.argmax(this_y) 
                ).item()


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
        
        loss_avg_feature_matching = loss_feat_total / args.syn_steps
        loss_avg_feature_step = loss_feat_step_total / args.syn_steps
        acc_avg_feat = acc_feat_total / args.syn_steps
        acc_avg_feat = f"{acc_avg_feat}%"

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        optimizer_feat.zero_grad()

        grand_loss.backward(retain_graph=True)
        loss_feat_total.backward()

        optimizer_img.step()
        optimizer_lr.step()
        optimizer_feat.step()

        # wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                #    "Start_Epoch": start_epoch})
        # if it % 50 == 0:
        #     # print('%s iter = %04d, Matching loss = %.4f' % (get_time(), it, loss_avg))
        #     print('{} it = {} Grand_Loss = {},Start_Epoch = {} '.format(get_time(),it,param_loss.detach().cpu(),start_epoch))
        #     print('loss_feat_total = {},Start_Epoch = {} '.format(loss_feat_total.detach().cpu(),start_epoch))
        #     print('feat matching loss = ', loss_avg_feature_matching)
        #     print('loss_avg_feature_step = ', loss_avg_feature_step)
        #     print('feat acc_avg_feat = ', acc_avg_feat)

        for _ in student_params:
            del _

        if it%10 == 0:
            print('{} it = {} Grand_Loss = {},Start_Epoch = {} '.format(get_time(),it,param_loss.detach().cpu(),start_epoch))
            print('loss_feat_total = {},Start_Epoch = {} '.format(loss_feat_total.detach().cpu(),start_epoch))
            print('feat matching loss = ', loss_avg_feature_matching)
            print('loss_avg_feature_step = ', loss_avg_feature_step)
            print('feat acc_avg_feat = ', acc_avg_feat)
    # 训练结束再落一份“最后”ckpt
    _save_ckpt(ckpt_last, args.Iteration, best_acc)

    # wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument("--lbd", type=float, default=0.01, help="scale of MSE")
    parser.add_argument(
        "--lr_feat",
        type=float,
        default=0.1,
        help="learning rate for updating synthetic features",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=None, help="layer index for feature"
    )
    parser.add_argument("--pooling", type=str, choices=["avg", "max", "sum", "none"], default="none",
                    help="feature pooling method (avg/max/sum/none)")
    parser.add_argument("--feat-lbd", type=float, default=0.02, help="scale of CE")
    parser.add_argument("--feat-opt", type=str, default="SGD", help="featureoptimizer")
    parser.add_argument("--n-feat", type=int, default=1, help="number of features")
    parser.add_argument("--eval-freq", type=int, default=4, help="evaluate frequency")
    parser.add_argument(
        "--use-feature", type=str, default="mean", help="the way to use the feature"
    )
    parser.add_argument(
        "--feat-norm", action="store_true", default=False, help="normalize the feature"
    )
    parser.add_argument(
        "--contrast",
        action="store_true",
        default=False,
        help="use the contrastive loss",
    )
    parser.add_argument(
        "--lbd-contrast", type=float, default=0.1, help="scale of contrastive loss"
    )
    parser.add_argument(
        "--feat-metric", type=str, default="MSE", help="feature criterion"
    )
    parser.add_argument(
        "--img-method", default="MTT", help="image distillation method"
    )
    parser.add_argument("--img-path", default="../distilled_data", help="image path")
    parser.add_argument("--res-path", default="res", help="result path")
    parser.add_argument('--generate_pretrained', action='store_true', default=False,
                    help='Generate images_best.pt from real data and exit')
    parser.add_argument('--resume_from', type=str, default='',
                    help='path to checkpoint to resume from (e.g., ./logged_files/.../ckpt_last.pt)')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save a checkpoint every N iterations')

    args = parser.parse_args()
    
    if args.dataset == 'ImageNet':
        model_args = f'{args.dataset}_{args.subset}_{args.ipc}ipc_[{args.feat_metric}{args.lbd}_pool{args.pooling}_layer{args.layer_idx}_CE{args.feat_lbd}_opt{args.feat_opt}_nfeat{args.n_feat}_use{args.use_feature}_norm{args.feat_norm}]_{args.img_method}_lr-feat{args.lr_feat}_{get_time()}'
    else:
        model_args = f'{args.dataset}_{args.ipc}ipc_[{args.feat_metric}{args.lbd}_pool{args.pooling}_layer{args.layer_idx}_CE{args.feat_lbd}_opt{args.feat_opt}_nfeat{args.n_feat}_use{args.use_feature}_norm{args.feat_norm}]_{args.img_method}_lr-feat{args.lr_feat}_{get_time()}'
    res_path = args.res_path
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    log_file = os.path.join(res_path, f'{model_args}.txt')
    log_file_handle = open(log_file, 'w')
    sys.stdout = FlushFile(log_file_handle)

    main(args)


