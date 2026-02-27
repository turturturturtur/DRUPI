from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


class AverageMeter:
    """Track running average of a scalar metric."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0
        self.val = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> List[torch.Tensor]:
    """Compute top-k accuracy for given predictions and targets."""

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res: List[torch.Tensor] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

