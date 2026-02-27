from __future__ import annotations

from typing import List

import torch.nn as nn


def get_parameters(model: nn.Module):
    """Split parameters into weight-decay and no-weight-decay groups."""
    group_no_weight_decay: List[nn.Parameter] = []
    group_weight_decay: List[nn.Parameter] = []
    for pname, p in model.named_parameters():
        if pname.find("weight") >= 0 and len(p.size()) > 1:
            group_weight_decay.append(p)
        else:
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(
        group_no_weight_decay
    )
    groups = [
        dict(params=group_weight_decay),
        dict(params=group_no_weight_decay, weight_decay=0.0),
    ]
    return groups

