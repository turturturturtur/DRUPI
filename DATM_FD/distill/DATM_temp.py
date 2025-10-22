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
from utils.utils_baseline import (
    get_dataset,
    get_network,
    get_eval_pool,
    evaluate_synset,
    get_time,
    DiffAugment,
    ParamDiffAug,
    TensorDataset,
    epoch,
    FlushFile,
    get_time,
    epoch_with_features,
    evaluate_synset_w_feature,
)
import wandb
import copy
import random
from reparam_module import ReparamModule

# from kmeans_pytorch import kmeans
from utils.cfg import CFG as cfg
import warnings
import yaml

from functools import partial
import matplotlib.pyplot as plt
from feature_metric import MMDLoss, CosineLoss, SupConLoss
import os.path as osp

os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）
# os.environ["WANDB_MODE"] = "disabled"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# redirect stdout to file



def manual_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    manual_seed()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in args.device])

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    #
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

    if args.skip_first_eva == False:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(
            args.eval_it, args.Iteration + 1, args.eval_it
        ).tolist()
    (
        channel,
        im_size,
        num_classes,
        class_names,
        mean,
        std,
        dst_train,
        dst_test,
        testloader,
        loader_train_dict,
        class_map,
        class_map_inv,
    ) = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args
    )
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict()  # record performances of all experiments
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

    wandb.init(
        sync_tensorboard=False,
        project="Feature_DATM",
        job_type="CleanRepo",
        entity='FD_distillation', 
        config=args,
        name="{}_{}ipc_[{}{}_pool{}_layer{}_CE{}_opt{}_nfeat{}_use{}_norm{}]_{}_{}".format(args.dataset,args.ipc,args.feat_metric,args.lbd,args.pooling,args.layer_idx,args.feat_lbd,args.feat_opt,args.n_feat,args.use_feature,args.feat_norm,args.img_method,get_time()),
    )
    args = type("", (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1

    print("Hyper-parameters: \n", args.__dict__)
    print("Evaluation model pool: ", model_eval_pool)

    """ organize the real dataset """
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    if (
        args.dataset == "ImageNet1K"
        and os.path.exists("images_all.pt")
        and os.path.exists("labels_all.pt")
    ):
        images_all = torch.load("images_all.pt")
        labels_all = torch.load("labels_all.pt")
    else:
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
        images_all = torch.cat(images_all, dim=0).to("cpu")
        labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
        if args.dataset == "ImageNet1K":
            torch.save(images_all, "images_all.pt")
            torch.save(labels_all, "labels_all.pt")

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)

    #
    # define pooling function
    pooling_function = None
    images_norm, features_norm = [], []
    if args.pooling:
        if args.pooling == "avg":
            # avgpooling2d function
            pooling_function = partial(
                torch.nn.functional.adaptive_avg_pool2d, output_size=(1, 1)
            )
        elif args.pooling == "max":
            # maxpooling2d function
            pooling_function = partial(
                torch.nn.functional.adaptive_max_pool2d, output_size=(1, 1)
            )
        elif args.pooling == "sum":
            # sumpooling2d function
            pooling_function = partial(
                torch.nn.functional.adaptive_sum_pool2d, output_size=(1, 1)
            )
        else:
            raise NotImplementedError("Pooling method not implemented")
    #

    for c in range(num_classes):
        print("class c = %d: %d real images" % (c, len(indices_class[c])))

    for ch in range(channel):
        print(
            "real images channel %d, mean = %.4f, std = %.4f"
            % (ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch]))
        )

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    """ initialize the synthetic data """
    label_syn = torch.tensor(
        [np.ones(args.ipc) * i for i in range(num_classes)],
        dtype=torch.long,
        requires_grad=False,
        device=args.device,
    ).view(
        -1
    )  # [0,0,0, 1,1,1, ..., 9,9,9]

    image_syn = torch.randn(
        size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
        dtype=torch.float,
    )

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
            buffer = buffer + torch.load(
                os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
            )
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(
                os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
            )
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        # random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[: args.max_files]

        expert_id = [i for i in range(len(expert_files))]
        random.shuffle(expert_id)

        print("loading file {}".format(expert_files[expert_id[file_idx]]))
        buffer = torch.load(expert_files[expert_id[file_idx]])
        if args.max_experts is not None:
            buffer = buffer[: args.max_experts]
        buffer_id = [i for i in range(len(buffer))]
        random.shuffle(buffer_id)

    if args.pix_init == "real":
        print("initialize synthetic data from pretrained data")
        image_syn_pretrained = torch.load(osp.join(load_path, f"images_best.pt")).to(
            args.device
        )
        image_syn = (
            image_syn_pretrained.clone().detach().requires_grad_(True).to(args.device)
        )
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
            optimizer_pretrained = torch.optim.SGD(
                pretrained_net.parameters(), lr=args.lr_teacher
            )
            optimizer_pretrained.zero_grad()
            criterion = nn.CrossEntropyLoss().to(args.device)
            epoch(
                "train",
                trainloader,
                pretrained_net,
                optimizer_pretrained,
                criterion,
                args,
                False,
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

    elif args.pix_init == "samples_predicted_correctly":
        if args.parall_eva == False:
            device = torch.device("cuda:0")
        else:
            device = args.device
        label_expert_files = expert_files
        image_syn_pretrained = torch.load(osp.join(load_path, f"images_best.pt")).to(
            args.device
        )
        image_syn = (
            image_syn_pretrained.clone().detach().requires_grad_(True).to(args.device)
        )
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
                False,
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
    else:
        print("initialize synthetic data from random noise")
        image_syn = torch.randn(
            size=(num_classes * args.ipc, channel, im_size[0], im_size[1]),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )
        feature_syn = torch.randn(
            size=(
                args.n_feat,
                num_classes * args.ipc,
                feat_size[0],
                feat_size[1],
                feat_size[2],
            ),
            dtype=torch.float,
            requires_grad=True,
            device=args.device,
        )

    """ training """
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

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

    optimizer_img.zero_grad()

    ###

    """test"""

    def SoftCrossEntropy(inputs, target, reduction="average"):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    criterion = SoftCrossEntropy

    print("%s training begins" % get_time())
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}



    label_syn_pretrained = torch.load(osp.join(load_path, f"labels_best.pt")).to(
            args.device
        )
    label_syn = (
            label_syn_pretrained.clone().detach().requires_grad_(True).to(args.device)
        )
    del label_syn_pretrained

    # print("label_syn: ",label_syn)
    # label_syn = label_syn.to(args.device)

    optimizer_y = torch.optim.SGD([label_syn], lr=args.lr_y, momentum=args.Momentum_y)
    vs = torch.zeros_like(label_syn)
    accumulated_grad = torch.zeros_like(label_syn)
    last_random = 0

    # del Temp_net

    # test
    curMax_times = 0
    current_accumulated_step = 0

    for it in range(0, args.Iteration + 1):
        save_this_it = False
        # wandb.log({"Progress": it}, step=it)
        """ Evaluate synthetic data """
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print(
                    "-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d"
                    % (args.model, model_eval, it)
                )
                if args.dsa:
                    print("DSA augmentation strategy: \n", args.dsa_strategy)
                    print("DSA augmentation parameters: \n", args.dsa_param.__dict__)
                else:
                    print("DC augmentation parameters: \n", args.dc_aug_param)

                accs_test = []
                accs_train = []
                #
                if it == 0:
                    feat_loss = None
                else:
                    feat_loss = loss_avg_feature_cls
                #
                for it_eval in range(args.num_eval):
                    if args.parall_eva == False:
                        device = torch.device("cuda:0")
                        net_eval = get_network(
                            model_eval, channel, num_classes, im_size, dist=False
                        ).to(
                            device
                        )  # get a random model
                    else:
                        device = args.device
                        net_eval = get_network(
                            model_eval, channel, num_classes, im_size, dist=True
                        ).to(
                            device
                        )  # get a random model
                    
                    eval_labs = label_syn.detach().to(device)
                    with torch.no_grad():
                        image_save = image_syn.to(device)
                    image_syn_eval, label_syn_eval = copy.deepcopy(
                        image_save.detach()
                    ).to(device), copy.deepcopy(eval_labs.detach()).to(
                        device
                    )  # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    feature_syn_eval = copy.deepcopy(feature_syn.detach())
                    if it_eval == 0:
                        print("Evaluate without feature!")
                        _, acc_train, acc_test = evaluate_synset(
                            it_eval,
                            copy.deepcopy(net_eval).to(device),
                            image_syn_eval.to(device),
                            label_syn_eval.to(device),
                            testloader,
                            args,
                            texture=False,
                            train_criterion=criterion,
                        )
                    else:
                        _, acc_train, acc_test = evaluate_synset_w_feature(
                            it_eval,
                            copy.deepcopy(net_eval).to(device),
                            image_syn_eval.to(device),
                            label_syn_eval.to(device),
                            feature_syn_eval,
                            testloader,
                            args,
                            feature_criterion,
                            texture=False,
                            train_criterion=criterion,
                            pooling_function=pooling_function,
                            feat_loss=feat_loss,
                            feature_strategy=feature_strategy,
                        )
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
                print(
                    "Evaluate %d random %s, mean = %.4f std = %.4f strategy = %s \n-------------------------"
                    % (len(accs_test), model_eval, acc_test_mean, acc_test_std,feature_strategy)
                )
                wandb.log({"Accuracy/{}".format(model_eval): acc_test_mean}, step=it)
                print('iter = {},model_eval = {}, Accuracy = {},'.format(it,model_eval,acc_test_mean))
                wandb.log(
                    {"Max_Accuracy/{}".format(model_eval): best_acc[model_eval]},
                    step=it,
                )
                print('iter = {},model_eval = {}, Max_Accuracy = {},'.format(it,model_eval, best_acc[model_eval]))
                wandb.log({"Std/{}".format(model_eval): acc_test_std}, step=it)
                print('iter = {},model_eval = {}, Std = {},'.format(it,model_eval,acc_test_std))
                wandb.log(
                    {"Max_Std/{}".format(model_eval): best_std[model_eval]}, step=it
                )
                print('iter = {},model_eval = {}, Max_Std = {},'.format(it,model_eval,best_std[model_eval]))

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()
                save_dir = os.path.join(
                    ".",
                    "logged_files",
                    args.dataset,
                    str(args.ipc),
                    args.model,
                    wandb.run.name,
                )

                if not os.path.exists(save_dir):
                    os.makedirs(os.path.join(save_dir, "Normal"))

                torch.save(
                    image_save.cpu(),
                    os.path.join(save_dir, "Normal", "images_{}.pt".format(it)),
                )
                torch.save(
                    label_syn.cpu(),
                    os.path.join(save_dir, "Normal", "labels_{}.pt".format(it)),
                )
                torch.save(
                    syn_lr.detach().cpu(),
                    os.path.join(save_dir, "Normal", "lr_{}.pt".format(it)),
                )

                if save_this_it:
                    torch.save(
                        image_save.cpu(),
                        os.path.join(save_dir, "Normal", "images_best.pt".format(it)),
                    )
                    torch.save(
                        label_syn.cpu(),
                        os.path.join(save_dir, "Normal", "labels_best.pt".format(it)),
                    )
                    torch.save(
                        syn_lr.detach().cpu(),
                        os.path.join(save_dir, "Normal", "lr_best.pt".format(it)),
                    )

                wandb.log(
                    {
                        "Pixels": wandb.Histogram(
                            torch.nan_to_num(image_syn.detach().cpu())
                        )
                    },
                    step=it,
                )

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(
                        upsampled, nrow=10, normalize=True, scale_each=True
                    )
                    wandb.log(
                        {
                            "Synthetic_Images": wandb.Image(
                                torch.nan_to_num(grid.detach().cpu())
                            )
                        },
                        step=it,
                    )
                    wandb.log(
                        {
                            "Synthetic_Pixels": wandb.Histogram(
                                torch.nan_to_num(image_save.detach().cpu())
                            )
                        },
                        step=it,
                    )

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(
                            image_save,
                            min=mean - clip_val * std,
                            max=mean + clip_val * std,
                        )
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(
                                upsampled, repeats=4, dim=2
                            )
                            upsampled = torch.repeat_interleave(
                                upsampled, repeats=4, dim=3
                            )
                        grid = torchvision.utils.make_grid(
                            upsampled, nrow=10, normalize=True, scale_each=True
                        )
                        wandb.log(
                            {
                                "Clipped_Synthetic_Images/std_{}".format(
                                    clip_val
                                ): wandb.Image(torch.nan_to_num(grid.detach().cpu()))
                            },
                            step=it,
                        )

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()
                        torch.save(
                            image_save.cpu(),
                            os.path.join(
                                save_dir, "Normal", "images_zca_{}.pt".format(it)
                            ),
                        )
                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(
                                upsampled, repeats=4, dim=2
                            )
                            upsampled = torch.repeat_interleave(
                                upsampled, repeats=4, dim=3
                            )
                        grid = torchvision.utils.make_grid(
                            upsampled, nrow=10, normalize=True, scale_each=True
                        )
                        wandb.log(
                            {
                                "Reconstructed_Images": wandb.Image(
                                    torch.nan_to_num(grid.detach().cpu())
                                )
                            },
                            step=it,
                        )
                        wandb.log(
                            {
                                "Reconstructed_Pixels": wandb.Histogram(
                                    torch.nan_to_num(image_save.detach().cpu())
                                )
                            },
                            step=it,
                        )
                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(
                                image_save,
                                min=mean - clip_val * std,
                                max=mean + clip_val * std,
                            )
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(
                                    upsampled, repeats=4, dim=2
                                )
                                upsampled = torch.repeat_interleave(
                                    upsampled, repeats=4, dim=3
                                )
                            grid = torchvision.utils.make_grid(
                                upsampled, nrow=10, normalize=True, scale_each=True
                            )
                            wandb.log(
                                {
                                    "Clipped_Reconstructed_Images/std_{}".format(
                                        clip_val
                                    ): wandb.Image(
                                        torch.nan_to_num(grid.detach().cpu())
                                    )
                                },
                                step=it,
                            )

        # wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        # print('iter = {}, Synthetic_LR = {},'.format(it,syn_lr.detach().cpu()))

        student_net = get_network(
            args.model, channel, num_classes, im_size, dist=False
        ).to(
            args.device
        )  # get a random model

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
                    buffer = buffer[: args.max_experts]
                random.shuffle(buffer_id)

        # Only match easy traj. in the early stage
        if args.Sequential_Generation:
            Upper_Bound = args.current_max_start_epoch + int(
                (args.max_start_epoch - args.current_max_start_epoch)
                * it
                / (args.expansion_end_epoch)
            )
            Upper_Bound = min(Upper_Bound, args.max_start_epoch)
        else:
            Upper_Bound = args.max_start_epoch

        start_epoch = np.random.randint(args.min_start_epoch, Upper_Bound)

        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        target_params = torch.cat(
            [p.data.to(args.device).reshape(-1) for p in target_params], 0
        )

        student_params = [
            torch.cat(
                [p.data.to(args.device).reshape(-1) for p in starting_params], 0
            ).requires_grad_(True)
        ]
        starting_params = torch.cat(
            [p.data.to(args.device).reshape(-1) for p in starting_params], 0
        )

        syn_images = image_syn
        y_hat = label_syn
        syn_feat = feature_syn

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
            this_feat = syn_feat[:,these_indices]
            optimizer_feat.zero_grad()
            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
            if args.distributed:
                forward_params = (
                    student_params[-1]
                    .unsqueeze(0)
                    .expand(torch.cuda.device_count(), -1)
                )
            else:
                forward_params = student_params[-1]

            criterion_feature = feature_criterion.to(args.device)

            this_feat_fixed = this_feat
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

            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)
            loss_ce = ce_loss
            loss_reg = criterion_feature(feature, this_feat)
            loss_syn = loss_ce + lbd * loss_reg
            grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[
                0
            ]

            # network
            student_params.append(student_params[-1] - syn_lr * grad)

            # update features
            
        
            if isinstance(student_net, torch.nn.DataParallel):
                this_feat_fixed = this_feat_fixed.squeeze(0)
                output_feat_syn = student_net.module.get_output_from_features(
                    this_feat_fixed, layer_idx=args.layer_idx
                )
            else:
                this_feat_fixed = this_feat_fixed.squeeze(0)
                output_feat_syn = student_net.get_output_from_features(
                    this_feat_fixed, layer_idx=args.layer_idx
                )
            loss_feat_step = criterion(output_feat_syn, this_y)
            loss_feat_step_total += loss_feat_step
            with torch.no_grad():
                acc_feat_total += torch.sum(
                    torch.argmax(output_feat_syn, dim=1) == torch.argmax(this_y, dim=1) 
                ).item()

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(
            student_params[-1], target_params, reduction="sum"
        )
        
        param_dist += torch.nn.functional.mse_loss(
            starting_params, target_params, reduction="sum"
        )

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

        optimizer_y.zero_grad()  
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        optimizer_feat.zero_grad()
    
        grand_loss.backward(retain_graph=True)
        loss_feat_total.backward()
        
        if grand_loss <= args.threshold:
            optimizer_y.step()
            optimizer_img.step()
            optimizer_lr.step()
            optimizer_feat.step()
        else:
            print("-----------------------------")
            wandb.log({"falts": start_epoch}, step=it)
            print('iter = {},falts = {} '.format(it,start_epoch))
            
            
        # optimizer_y.zero_grad()  
        # optimizer_img.zero_grad()
        # optimizer_lr.zero_grad()
        # optimizer_feat.zero_grad()
    
        if it % 50 == 0:
            # print('%s iter = %04d, Matching loss = %.4f' % (get_time(), it, loss_avg))
            print('{} it = {} Grand_Loss = {},Start_Epoch = {} '.format(get_time(),it,param_loss.detach().cpu(),start_epoch))
            print('loss_feat_total = {},Start_Epoch = {} '.format(loss_feat_total.detach().cpu(),start_epoch))
            print('feat matching loss = ', loss_avg_feature_matching)
            print('loss_avg_feature_step = ', loss_avg_feature_step)
            print('feat acc_avg_feat = ', acc_avg_feat)
            

        wandb.log({"Grand_Loss": param_loss.detach().cpu(), "Start_Epoch": start_epoch})
        
        wandb.log({"loss_feat_total": loss_feat_total.detach().cpu(), "Start_Epoch": start_epoch})
        

        for _ in student_params:
            del _

        if it % 10 == 0:
            print("%s iter = %04d, grand_loss = %.4f, loss_feat_total = %.4f" % (get_time(), it, grand_loss.item(),loss_feat_total.item()))
            # print("%s iter = %04d, loss_feat_total = %.4f" % (get_time(), it, loss_feat_total.item()))

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")

    parser.add_argument("--cfg", type=str, default="")
    #
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
    parser.add_argument(
        "--pooling", type=str, default=None, help="feature pooling method"
    )
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
        "--img-method", default="DATM", help="image distillation method"
    )
    parser.add_argument("--img-path", default="../../distilled_data", help="image path")
    #
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = "--" + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()
    model_args = f'{args.dataset}_{args.ipc}ipc_[{args.feat_metric}{args.lbd}_pool{args.pooling}_layer{args.layer_idx}_CE{args.feat_lbd}_opt{args.feat_opt}_nfeat{args.n_feat}_use{args.use_feature}_norm{args.feat_norm}]_{args.img_method}_{get_time()}'

    log_file = os.path.join('res', f'{model_args}.txt')
    log_file_handle = open(log_file, 'w')
    sys.stdout = FlushFile(log_file_handle)
    
    main(args)
