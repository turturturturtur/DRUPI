import os
import sys

sys.path.append("../")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.utils_baseline import (
    get_dataset,
    get_network,
    get_eval_pool,
    ParamDiffAug,
    TensorDataset,
    epoch,
    get_time,
)
import copy
from reparam_module import ReparamModule
import time
import math
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from score_distillation import get_score_distillation_loss, get_ddpm


def optimize_images(
    net,
    images_train,
    labels_train,
    train_criterion,
    device,
    pipe,
    num_iterations=1000,
    learning_rate=0.01,
    lbd=0.1,
):
    # Set the model to evaluation mode
    net.eval()

    # Create a noise tensor and set it to require gradients
    noise = torch.zeros_like(images_train, requires_grad=True, device=device)

    ################################# get diffusion model #################################
    pipe.to(args.device)

    # Define the optimizer to optimize the noise
    optimizer = optim.Adam([noise], lr=learning_rate)

    for i in range(num_iterations):
        optimizer.zero_grad()
        # Add the noise to the images
        noisy_images = images_train + noise
        # Forward pass
        outputs = net(noisy_images)
        # Compute loss
        loss_sd = get_score_distillation_loss(pipe, noisy_images, steps=args.step)
        loss = train_criterion(outputs, labels_train) + lbd * loss_sd
        # Backward pass
        loss.backward()
        # Update the noise using the optimizer
        optimizer.step()

        if i == num_iterations // 2:
            for g in optimizer.param_groups:
                g["lr"] = learning_rate / 10
        if i % (num_iterations // 10) == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

    # Return the images with optimized noise
    optimized_images = images_train + noise
    return optimized_images


def evaluate(
    it_eval,
    net,
    images_train,
    labels_train,
    testloader,
    args,
    return_loss=False,
    texture=False,
    train_criterion=None,
    Preciser_Scheduler=False,
    type=1,
):
    if args.parall_eva == False:
        device = torch.device("cuda:1")
    else:
        device = args.device
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)

    if Preciser_Scheduler:
        LR_begin = 0.0000000001
        LR_End = float(args.lr_net)
        if type == 0:
            t = 0
        else:
            t = 500
        T = Epoch
        lambda1 = lambda epoch: (
            ((LR_End - LR_begin) * epoch / t)
            if epoch < t
            else LR_End * (1 + math.cos(math.pi * (epoch - t) / (T - t))) / 2.0
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=LR_End, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        lr_schedule = [Epoch // 2 + 1]
        optimizer = torch.optim.SGD(
            net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )

    test_criterion = nn.CrossEntropyLoss().to(device)
    If_Float = True
    if train_criterion == None:
        train_criterion = nn.CrossEntropyLoss().to(device)
        If_Float = False

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(
        dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0
    )

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    """train"""
    for ep in tqdm(range(Epoch + 1)):
        loss_train, acc_train = epoch(
            "train",
            trainloader,
            net,
            optimizer,
            train_criterion,
            args,
            aug=True,
            texture=texture,
            If_Float=If_Float,
        )
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)

        if Preciser_Scheduler:
            scheduler.step()
        else:
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(
                    net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
                )
    """test"""
    with torch.no_grad():
        loss_test, acc_test = epoch(
            "test",
            testloader,
            net,
            optimizer,
            test_criterion,
            args,
            aug=False,
            If_Float=False,
        )
    time_train = time.time() - start

    print(
        "Before Synthesis: %s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
        % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test)
    )
    for lbd in [1, 0.1, 0.01, 0.001, 10]:
        model_path = f"/mnt/workspace/wangshaobo/SDDD/ddpm_ema_cifar10"
        pipe = get_ddpm(model_path)
        """Optimize the entire image dataset"""
        images_train_w_noise = optimize_images(
            net, images_train, labels_train, train_criterion, device, pipe, lbd=lbd
        )

        start = time.time()
        acc_train_list = []
        loss_train_list = []

        dst_train = TensorDataset(images_train_w_noise, labels_train)
        trainloader = torch.utils.data.DataLoader(
            dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0
        )
        """train"""
        for ep in tqdm(range(Epoch + 1)):
            loss_train, acc_train = epoch(
                "train",
                trainloader,
                net,
                optimizer,
                train_criterion,
                args,
                aug=True,
                texture=texture,
                If_Float=If_Float,
            )
            acc_train_list.append(acc_train)
            loss_train_list.append(loss_train)

            if Preciser_Scheduler:
                scheduler.step()
            else:
                if ep in lr_schedule:
                    lr *= 0.1
                    optimizer = torch.optim.SGD(
                        net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
                    )
        """test"""
        with torch.no_grad():
            loss_test, acc_test = epoch(
                "test",
                testloader,
                net,
                optimizer,
                test_criterion,
                args,
                aug=False,
                If_Float=False,
            )

        time_train = time.time() - start

        print(
            "%s With lambda %.3f, Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f"
            % (
                get_time(),
                lbd,
                it_eval,
                Epoch,
                int(time_train),
                loss_train,
                acc_train,
                acc_test,
            )
        )

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test


def main(args):

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == "True" else False
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

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

    args.im_size = im_size

    if args.dsa:
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    print("Hyper-parameters: \n", args.__dict__)
    print("Evaluation model pool: ", model_eval_pool)

    def SoftCrossEntropy(inputs, target, reduction="average"):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    soft_cri = SoftCrossEntropy

    image_syn_eval = torch.load(args.data_dir)
    label_syn_eval = torch.load(args.label_dir)
    args.lr_net = torch.load(args.lr_dir)

    for model_eval in model_eval_pool:
        print("Evaluating: " + model_eval)
        network = get_network(model_eval, channel, num_classes, im_size, dist=False).to(
            args.device
        )  # get a random model
        _, acc_train, acc_test = evaluate(
            0,
            copy.deepcopy(network),
            image_syn_eval,
            label_syn_eval,
            testloader,
            args,
            texture=False,
            train_criterion=soft_cri,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset")
    parser.add_argument(
        "--subset",
        type=str,
        default="imagenette",
        help="ImageNet subset. This only does anything when --dataset=ImageNet",
    )
    parser.add_argument("--model", type=str, default="ConvNet", help="model")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="S",
        help="eval_mode, check utils.py for more info",
    )
    parser.add_argument(
        "--epoch_eval_train",
        type=int,
        default=1000,
        help="epochs to train a model with synthetic data",
    )
    parser.add_argument(
        "--batch_real", type=int, default=256, help="batch size for real data"
    )
    parser.add_argument(
        "--dsa",
        type=str,
        default="True",
        choices=["True", "False"],
        help="whether to use differentiable Siamese augmentation.",
    )
    parser.add_argument(
        "--dsa_strategy",
        type=str,
        default="color_crop_cutout_flip_scale_rotate",
        help="differentiable Siamese augmentation strategy",
    )
    parser.add_argument("--data_path", type=str, default="data", help="dataset path")
    parser.add_argument("--zca", action="store_true", help="do ZCA whitening")
    parser.add_argument(
        "--lr_teacher",
        type=float,
        default=0.01,
        help="initialization for synthetic learning rate",
    )
    parser.add_argument(
        "--no_aug",
        type=bool,
        default=False,
        help="this turns off diff aug during distillation",
    )
    parser.add_argument(
        "--batch_train", type=int, default=128, help="batch size for training networks"
    )

    parser.add_argument("--parall_eva", type=bool, default=False, help="dataset")

    parser.add_argument("--data_dir", type=str, default="path", help="dataset")
    parser.add_argument("--label_dir", type=str, default="path", help="dataset")
    parser.add_argument("--lr_dir", type=str, default="path", help="dataset")
    parser.add_argument("--step", type=int, default=10, help="dataset")

    args = parser.parse_args()

    main(args)
