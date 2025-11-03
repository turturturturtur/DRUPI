"""
完整的、自包含的DATM.py版本。
- 彻底移除了对Weights & Biases (wandb)的所有依赖。
- 将IPC48.yaml中的所有配置参数直接整合到了argparse中作为默认值。
- 修复了因wandb.run.name为None而导致的路径拼接TypeError。
- 现在可以通过命令行直接运行，无需--cfg参数。
"""

import os
import sys
sys.path.append("../")  # 确保可以找到utils文件夹
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import copy
import random
from reparam_module import ReparamModule
import datetime  # 用于生成唯一的运行名称

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def manual_seed(seed=0):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保cudnn的确定性，可能会轻微影响性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    """主训练函数"""
    manual_seed()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.device])

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- WANDB移除: 创建一个唯一的运行名称来替代wandb.run.name ---
    run_name = f"{args.dataset}_{args.model}_ipc{args.ipc}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"本次运行的唯一名称 (Run Name): {run_name}")
    # ---

    if not args.skip_first_eva:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    args.im_size = im_size

    if args.dsa:
        args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc
    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    # --- 数据集组织 ---
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    # (此部分逻辑保持不变)
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    # --- 合成数据初始化 ---
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    # --- 专家轨迹 Buffer 加载 ---
    # (此部分逻辑保持不变)
    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))
    # (加载逻辑保持不变)
    expert_files = []
    n = 0
    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))
    file_idx = 0
    expert_idx = 0
    if args.max_files is not None:
        expert_files = expert_files[:args.max_files]
    expert_id = list(range(len(expert_files)))
    random.shuffle(expert_id)
    print("loading file {}".format(expert_files[expert_id[file_idx]]))
    buffer = torch.load(expert_files[expert_id[file_idx]])
    if args.max_experts is not None:
        buffer = buffer[:args.max_experts]
    buffer_id = list(range(len(buffer)))
    random.shuffle(buffer_id)
    
    # --- 根据 pix_init 初始化合成图像 ---
    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    # --- 训练准备 ---
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        return torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
    criterion = SoftCrossEntropy

    print('%s training begins'%get_time())
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    # --- 合成标签初始化 ---
    if args.pix_init == "real":
        Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        Temp_params = buffer[0][-1]
        Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
        if args.distributed:
            Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        image_syn_on_device = image_syn.to(args.device)    
        Initialized_Labels = Temp_net(image_syn, flat_param=Initialize_Labels_params)
        acc = np.sum(np.equal(np.argmax(Initialized_Labels.cpu().data.numpy(), axis=-1), label_syn.cpu().data.numpy()))
        print('InitialAcc (before softmax):{}'.format(acc/len(label_syn)))
        label_syn = copy.deepcopy(Initialized_Labels.detach()).to(args.device).requires_grad_(True)
        del Temp_net
    else: # 默认使用one-hot初始化
        label_syn = F.one_hot(label_syn, num_classes=num_classes).float().requires_grad_(True)

    optimizer_y = torch.optim.SGD([label_syn], lr=args.lr_y, momentum=args.Momentum_y)

    # --- 主训练循环 ---
    for it in range(args.Iteration + 1):
        save_this_it = False
        
        # --- 评估阶段 ---
        if it in eval_it_pool:
            print(f'\n================== Evaluation at Iteration {it} ==================')
            for model_eval in model_eval_pool:
                print('--- Evaluating model: %s ---' % model_eval)
                accs_test = []
                for it_eval in range(args.num_eval):
                    # device_eval = torch.device("cuda:0") if not args.parall_eva else args.device
                    device_eval = args.device
                    net_eval = get_network(model_eval, channel, num_classes, im_size, dist=False).to(device_eval)
                    
                    with torch.no_grad():
                        image_syn_eval, label_syn_eval = image_syn.detach(), label_syn.detach()

                    args.lr_net = syn_lr.item()
                    acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, train_criterion=criterion)
                    accs_test.append(acc_test)
                
                accs_test = np.array(accs_test)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)

                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                
                print('Evaluate %d random %s, mean = %.4f std = %.4f' % (len(accs_test), model_eval, acc_test_mean, acc_test_std))
                print('Best accuracy so far: %.4f' % best_acc[model_eval])
                print('------------------------------------------')

        # --- 保存阶段 ---
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                # --- WANDB移除 & 路径修复 ---
                save_dir = os.path.join(".", "logged_files", args.dataset, str(args.ipc), args.model, run_name)
                # ---
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                torch.save(image_syn.cpu(), os.path.join(save_dir, f"images_{it}.pt"))
                torch.save(label_syn.cpu(), os.path.join(save_dir, f"labels_{it}.pt"))
                torch.save(syn_lr.cpu(), os.path.join(save_dir, f"lr_{it}.pt"))
                print(f"Saved synthetic data at iteration {it} to {save_dir}")

                if save_this_it:
                    torch.save(image_syn.cpu(), os.path.join(save_dir, "images_best.pt"))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt"))
                    torch.save(syn_lr.cpu(), os.path.join(save_dir, "lr_best.pt"))
                    print(f"Saved BEST synthetic data at iteration {it} to {save_dir}")

        # --- 轨迹匹配和梯度更新 ---
        student_net = ReparamModule(get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device))
        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)
        student_net.train()
        num_params = sum([np.prod(p.size()) for p in student_net.parameters()])

        expert_trajectory = buffer[buffer_id[expert_idx]]
        expert_idx = (expert_idx + 1) % len(buffer)
        if expert_idx == 0: # 如果一个buffer用完了，换下一个文件
            file_idx = (file_idx + 1) % len(expert_files)
            print(f"Loading next expert file: {expert_files[expert_id[file_idx]]}")
            del buffer
            buffer = torch.load(expert_files[expert_id[file_idx]])
            if args.max_experts is not None: buffer = buffer[:args.max_experts]
            buffer_id = list(range(len(buffer))); random.shuffle(buffer_id)

        start_epoch = np.random.randint(args.min_start_epoch, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
        student_params = [starting_params.requires_grad_(True)]
        
        for step in range(args.syn_steps):
            indices = torch.randperm(len(image_syn))
            indices_chunks = list(torch.split(indices, args.batch_syn))
            x = image_syn[indices_chunks[0]]
            this_y = label_syn[indices_chunks[0]]

            if args.dsa:
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            forward_params = student_params[-1]
            if args.distributed:
                forward_params = forward_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)

            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)
            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)

        param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        grand_loss = param_loss / (param_dist + 1e-8) # 避免除以零
        
        optimizer_img.zero_grad()
        optimizer_y.zero_grad()
        optimizer_lr.zero_grad()
        grand_loss.backward()
        
        if grand_loss.item() <= args.threshold:
            optimizer_img.step()
            optimizer_y.step()
            optimizer_lr.step()
        
        if it % 10 == 0:
            print(f"{get_time()} Iter: {it:04d}, Grand Loss: {grand_loss.item():.4f}, Start Epoch: {start_epoch}, Syn LR: {syn_lr.item():.5f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    
    # --- 整合 IPC48.yaml 中的所有参数作为默认值 ---
    
    # 基本设置
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=48, help='images per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval mode')
    parser.add_argument('--num_eval', type=int, default=3, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=500, help='how often to evaluate')
    parser.add_argument('--skip_first_eva', action='store_true', default=False, help='if skip the first evaluation')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=10000, help='how many distillation steps to perform')
    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_y', type=float, default=10.0, help='learning rate for updating synthetic labels')
    parser.add_argument('--Momentum_y', type=float, default=0.9, help='momentum for updating synthetic labels')
    parser.add_argument('--threshold', type=float, default=1.0, help='threshold for training')
    parser.add_argument('--pix_init', type=str, default='real', help='initialize synthetic images from real images')
    parser.add_argument('--zca', action='store_true', default=True, help='do ZCA whitening')
    
    # 路径和设备
    parser.add_argument('--data_path', type=str, default='../dataset', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='../buffer_storage/', help='buffer path')
    parser.add_argument('--device', nargs='+', type=int, default=[0], help='GPU devices') # 默认使用GPU 0
    
    # 轨迹匹配参数
    parser.add_argument('--syn_steps', type=int, default=80, help='how many steps to take on synthetic data')
    parser.add_argument('--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--max_start_epoch', type=int, default=70, help='max epoch we can start at')
    parser.add_argument('--current_max_start_epoch', type=int, default=30, help='current max epoch we can start at, for sequential generation')
    parser.add_argument('--min_start_epoch', type=int, default=20, help='min epoch we can start at')
    
    # 批次大小
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--batch_syn', type=int, default=1000, help='batch size for syn loader')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    
    # 其他
    parser.add_argument('--dsa', type=str, default='True', help='differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='dsa strategy')
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', action='store_true', help='this turns off diff aug during distillation')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to use')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to use')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--parall_eva', action='store_true', default=False)
    parser.add_argument('--Sequential_Generation', action='store_true', default=False, help='Sequential Generation')

    args = parser.parse_args()
    main(args)
