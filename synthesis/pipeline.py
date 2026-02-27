from __future__ import annotations

import os
import shutil
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

from augmentation import denormalize
from config import ExperimentConfig
from data import build_raw_train_loader
from models import build_teacher


def _pad(input_tensor: torch.Tensor, target_height: int, target_width: Optional[int] = None):
    """Pad tensor with zeros to the target size."""
    if target_width is None:
        target_width = target_height
    vertical_padding = target_height - input_tensor.size(2)
    horizontal_padding = target_width - input_tensor.size(3)

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    padded_tensor = F.pad(
        input_tensor,
        (left_padding, right_padding, top_padding, bottom_padding),
    )
    return padded_tensor


def _batched_forward(model: nn.Module, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    total_samples = tensor.size(0)
    all_outputs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = tensor[i : min(i + batch_size, total_samples)]
            output = model(batch_data)
            all_outputs.append(output)

    final_output = torch.cat(all_outputs, dim=0)
    return final_output


def _cross_entropy(y_pre: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_pre = F.softmax(y_pre, dim=1)
    return (-torch.log(y_pre.gather(1, y.view(-1, 1))))[:, 0]


def _selector(
    n: int,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    size: int,
    m: int = 5,
) -> torch.Tensor:
    """Select top-n informative crops using teacher predictions."""
    with torch.no_grad():
        # [mipc, m, 3, H, W]
        images = images.cuda()
        s = images.shape

        # [mipc * m, 3, H, W]
        images = images.permute(1, 0, 2, 3, 4)
        images = images.reshape(s[0] * s[1], s[2], s[3], s[4])

        # [mipc * m]
        labels = labels.repeat(m).cuda()

        # [mipc * m, n_class]
        batch_size = s[0]  # Change it for small GPU memory
        preds = _batched_forward(model, _pad(images, size).cuda(), batch_size)

        # [mipc * m]
        dist = _cross_entropy(preds, labels)

        # [m, mipc]
        dist = dist.reshape(m, s[0])

        # [mipc]
        index = torch.argmin(dist, 0)
        dist = dist[index, torch.arange(s[0])]

        # [mipc, 3, H, W]
        sa = images.shape
        images = images.reshape(m, s[0], sa[1], sa[2], sa[3])
        images = images[index, torch.arange(s[0])]

    indices = torch.argsort(dist, descending=False)[:n]
    torch.cuda.empty_cache()
    return images[indices].detach()


def _mix_images(input_img: torch.Tensor, out_size: int, factor: int, n: int) -> torch.Tensor:
    """Mix small patches into a single large image grid."""
    s = out_size // factor
    remained = out_size % factor
    k = 0
    mixed_images = torch.zeros(
        (n, 3, out_size, out_size),
        requires_grad=False,
        dtype=torch.float,
    )
    h_loc = 0
    for i in range(factor):
        h_r = s + 1 if i < remained else s
        w_loc = 0
        for j in range(factor):
            w_r = s + 1 if j < remained else s
            img_part = F.interpolate(
                input_img.data[k * n : (k + 1) * n],
                size=(h_r, w_r),
            )
            mixed_images.data[
                0:n,
                :,
                h_loc : h_loc + h_r,
                w_loc : w_loc + w_r,
            ] = img_part
            w_loc += w_r
            k += 1
        h_loc += h_r
    return mixed_images


def _save_images(cfg: ExperimentConfig, images: torch.Tensor, class_id: int) -> None:
    images = denormalize(images)
    for idx in range(images.shape[0]):
        dir_path = f"{cfg.syn_data_path}/{class_id:05d}"
        os.makedirs(dir_path, exist_ok=True)
        place_to_store = f"{dir_path}/class{class_id:05d}_id{idx:05d}.jpg"
        image_np = images[idx].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def _extract_and_save_features(
    teacher: nn.Module,
    images: torch.Tensor,
    class_id: int,
    syn_data_path: str,
) -> None:
    """Extract teacher penultimate features from mixed images and save as .pt."""
    model = teacher.module if hasattr(teacher, "module") else teacher
    model.eval()
    with torch.no_grad():
        _, features = model(images.cuda(), return_features=True)
    dir_path = f"{syn_data_path}/{class_id:05d}"
    os.makedirs(dir_path, exist_ok=True)
    torch.save(features.cpu(), os.path.join(dir_path, "feat_labels.pt"))


def run_synthesis(cfg: ExperimentConfig) -> None:
    """Run the synthesis phase to generate distilled images."""

    print("Running synthesis with configuration:")
    print(cfg.asdict())

    # prepare output directory
    if os.path.exists(cfg.syn_data_path):
        shutil.rmtree(cfg.syn_data_path)
    os.makedirs(cfg.syn_data_path, exist_ok=True)

    # build teacher model
    teacher = build_teacher(cfg)
    teacher = nn.DataParallel(teacher).cuda()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # build data loader over raw training set
    train_loader = build_raw_train_loader(cfg)

    with torch.no_grad():
        for class_id, (images, labels) in enumerate(tqdm(train_loader)):
            selected = _selector(
                cfg.ipc * cfg.factor ** 2,
                teacher,
                images,
                labels,
                cfg.input_size,
                m=cfg.num_crop,
            )
            mixed = _mix_images(
                selected,
                cfg.input_size,
                cfg.factor,
                cfg.ipc,
            )
            _save_images(cfg, mixed, class_id)
            if cfg.use_feat_labels:
                _extract_and_save_features(
                    teacher, mixed, class_id, cfg.syn_data_path
                )

