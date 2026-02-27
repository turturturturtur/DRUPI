import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class MultiRandomCrop(torch.nn.Module):
    """Apply RandomResizedCrop multiple times to generate patches from an image."""

    def __init__(self, num_crop: int = 5, size: int = 224, factor: int = 2) -> None:
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

denormalize = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)


class ShufflePatches(torch.nn.Module):
    """Shuffle image patches along height and width."""

    def __init__(self, factor: int) -> None:
        super().__init__()
        self.factor = factor

    def _shuffle_one_dim(self, img: torch.Tensor, factor: int) -> torch.Tensor:
        h, w = img.shape[1:]
        th, tw = h // factor, w // factor
        patches = []
        for i in range(factor):
            start = i * tw
            if i != factor - 1:
                patches.append(img[..., start : start + tw])
            else:
                patches.append(img[..., start:])
        random.shuffle(patches)
        img = torch.cat(patches, -1)
        return img

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = self._shuffle_one_dim(img, self.factor)
        img = img.permute(0, 2, 1)
        img = self._shuffle_one_dim(img, self.factor)
        img = img.permute(0, 2, 1)
        return img


def rand_bbox(size: Tuple[int, int, int, int], lam: float):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(images: torch.Tensor, alpha: float):
    """Apply CutMix augmentation."""
    rand_index = torch.randperm(images.size()[0]).to(images.device)
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    return images, rand_index.cpu(), lam, [bbx1, bby1, bbx2, bby2]


def mixup(images: torch.Tensor, alpha: float):
    """Apply Mixup augmentation."""
    rand_index = torch.randperm(images.size()[0]).to(images.device)
    lam = np.random.beta(alpha, alpha)
    mixed_images = lam * images + (1 - lam) * images[rand_index]
    return mixed_images, rand_index.cpu(), lam, None


def mix_aug(images: torch.Tensor, mix_type: str, mixup_alpha: float, cutmix_alpha: float):
    """Dispatch to mixup or cutmix based on configuration."""
    if mix_type == "mixup":
        return mixup(images, mixup_alpha)
    if mix_type == "cutmix":
        return cutmix(images, cutmix_alpha)
    return images, None, None, None

