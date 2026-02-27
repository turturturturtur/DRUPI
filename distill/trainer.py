from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from augmentation import mix_aug
from config import ExperimentConfig
from data import build_synth_train_loader, build_val_loader
from models import build_student, build_teacher
from utils import AverageMeter, accuracy
from utils.optim import get_parameters


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def _set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


@dataclass
class TrainingResult:
    best_acc1: float
    best_epoch: int


class DistillationTrainer:
    """KL distillation trainer using synthesized data and real validation data."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

        if cfg.seed is not None:
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        print(f"=> using pytorch pre-trained teacher model '{cfg.arch_name}'")
        teacher_model = build_teacher(cfg)
        student_model = build_student(cfg)

        self.teacher_model = torch.nn.DataParallel(teacher_model).cuda()
        self.student_model = torch.nn.DataParallel(student_model).cuda()

        self.teacher_model.eval()
        self.student_model.train()

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        cudnn.benchmark = True

        # optimizer
        if cfg.sgd:
            self.optimizer = torch.optim.SGD(
                get_parameters(self.student_model),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                get_parameters(self.student_model),
                lr=cfg.adamw_lr,
                betas=[0.9, 0.999],
                weight_decay=cfg.adamw_weight_decay,
            )

        # lr scheduler
        if cfg.cos:
            self.scheduler = LambdaLR(
                self.optimizer,
                lambda step: 0.5
                * (1.0 + math.cos(math.pi * step / cfg.re_epochs / 2))
                if step <= cfg.re_epochs
                else 0,
                last_epoch=-1,
            )
        else:
            self.scheduler = LambdaLR(
                self.optimizer,
                lambda step: (1.0 - step / cfg.re_epochs)
                if step <= cfg.re_epochs
                else 0,
                last_epoch=-1,
            )

        print(f"process data from {cfg.syn_data_path}")

        # data loaders
        self.train_loader = build_synth_train_loader(cfg)
        self.val_loader = build_val_loader(cfg)

        print("load data successfully")

    def train_one_epoch(self, epoch: int) -> None:
        cfg = self.cfg
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        loss_function_kl = nn.KLDivLoss(reduction="batchmean")
        self.teacher_model.eval()
        self.student_model.train()

        t1 = time.time()
        for batch_idx, batch in enumerate(self.train_loader):
            # --- Unpack batch (2-tuple or 3-tuple with feature labels) ---
            if cfg.use_feat_labels:
                images, labels, feat_labels = batch
            else:
                images, labels = batch
                feat_labels = None

            # --- No-grad block: teacher forward + augmentation ---
            with torch.no_grad():
                images = images.cuda()
                labels = labels.cuda()
                if feat_labels is not None:
                    feat_labels = feat_labels.cuda()

                mix_images, _, _, _ = mix_aug(
                    images,
                    cfg.mix_type,
                    cfg.mixup,
                    cfg.cutmix,
                )

                soft_mix_label = self.teacher_model(mix_images)
                soft_mix_label = F.softmax(soft_mix_label / cfg.temperature, dim=1)

            if batch_idx % cfg.re_accum_steps == 0:
                self.optimizer.zero_grad()

            # --- Student forward on images (with features when DRUPI is active) ---
            if feat_labels is not None:
                pred_label, student_features = self.student_model(
                    images, return_features=True
                )
            else:
                with torch.no_grad():
                    pred_label = self.student_model(images)

            prec1, prec5 = accuracy(pred_label.detach(), labels, topk=(1, 5))

            # --- KL loss on augmented images ---
            pred_mix_label = self.student_model(mix_images)

            soft_pred_mix_label = F.log_softmax(
                pred_mix_label / cfg.temperature, dim=1
            )
            loss_kl = loss_function_kl(soft_pred_mix_label, soft_mix_label)

            # --- DRUPI privileged losses ---
            if feat_labels is not None:
                # L_reg: MSE between teacher feature labels and student features
                loss_reg = F.mse_loss(student_features, feat_labels)

                # L_task: teacher features through student classifier → CE
                student_classifier = self.student_model.module.classifier
                task_logits = student_classifier(feat_labels)
                loss_task = F.cross_entropy(task_logits, labels)

                loss = loss_kl + cfg.lambda_reg * loss_reg + cfg.lambda_task * loss_task
            else:
                loss = loss_kl

            loss = loss / cfg.re_accum_steps

            loss.backward()
            if batch_idx % cfg.re_accum_steps == (cfg.re_accum_steps - 1):
                self.optimizer.step()

            n = images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        print_info = (
            f"TRAIN Iter {epoch}: loss = {objs.avg:.6f},\t"
            + f"Top-1 err = {100 - top1.avg:.6f},\t"
            + f"Top-5 err = {100 - top5.avg:.6f},\t"
            + f"train_time = {time.time() - t1:.6f}"
        )
        print(print_info)

    def evaluate(self, epoch: Optional[int] = None) -> float:
        cfg = self.cfg
        objs = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss_function = nn.CrossEntropyLoss()

        self.student_model.eval()
        t1 = time.time()
        with torch.no_grad():
            for data, target in self.val_loader:
                target = target.type(torch.LongTensor)
                data, target = data.cuda(), target.cuda()

                output = self.student_model(data)
                loss = loss_function(output, target)

                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                n = data.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

        log_info = (
            f"TEST:\nIter {epoch}: loss = {objs.avg:.6f},\t"
            + f"Top-1 err = {100 - top1.avg:.6f},\t"
            + f"Top-5 err = {100 - top5.avg:.6f},\t"
            + f"val_time = {time.time() - t1:.6f}"
        )
        print(log_info)
        return float(top1.avg)

    def fit(self) -> TrainingResult:
        cfg = self.cfg
        best_acc1 = 0.0
        best_epoch = 0

        for epoch in range(cfg.re_epochs):
            self.train_one_epoch(epoch)

            if epoch % 10 == 9 or epoch == cfg.re_epochs - 1:
                if epoch > cfg.re_epochs * 0.8:
                    top1 = self.evaluate(epoch)
                else:
                    top1 = 0.0
            else:
                top1 = 0.0

            self.scheduler.step()
            if top1 > best_acc1:
                best_acc1 = max(top1, best_acc1)
                best_epoch = epoch

        print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")
        return TrainingResult(best_acc1=best_acc1, best_epoch=best_epoch)


def run_distillation(cfg: ExperimentConfig) -> TrainingResult:
    """Convenience function to run full distillation."""
    trainer = DistillationTrainer(cfg)
    return trainer.fit()

