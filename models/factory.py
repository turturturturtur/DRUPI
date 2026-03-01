from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torchvision.models as thmodels
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

from config import ExperimentConfig
from models.convnet import ConvNet


def _resolve_model_name_for_dataset(model_name: str, dataset: str) -> str:
    """Normalize model names to dataset-compatible variants."""
    if dataset in {"cifar10", "cifar100", "tinyimagenet"} and model_name == "resnet18":
        return "resnet18_modified"
    return model_name


class _FeatureExtractorWrapper(nn.Module):
    """Wrap a backbone to uniformly support ``forward(x, return_features=True)``
    and expose a ``.classifier`` property pointing to the final linear layer.

    This lets ResNet, VGG, EfficientNet, etc. behave like ConvNet for DRUPI.
    """

    def __init__(self, model: nn.Module, classifier_attr: str):
        super().__init__()
        self.model = model
        self._classifier_attr = classifier_attr

    @property
    def classifier(self) -> nn.Module:
        return getattr(self.model, self._classifier_attr)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if not return_features:
            return self.model(x)

        captured: dict = {}

        def _hook(_module, inp, _output):
            captured["feat"] = inp[0]

        handle = self.classifier.register_forward_hook(_hook)
        logits = self.model(x)
        handle.remove()
        return logits, captured["feat"]


def _ensure_feature_api(model: nn.Module) -> nn.Module:
    """Wrap non-ConvNet models so they support return_features and .classifier."""
    if isinstance(model, ConvNet):
        return model

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        clf_attr = "fc"
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        clf_attr = "classifier"
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        clf_attr = "head"
    else:
        return model

    return _FeatureExtractorWrapper(model, clf_attr)


def _patch_weights_enum() -> None:
    """Patch WeightsEnum.get_state_dict to be compatible with older weights."""

    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash", None)
        return load_state_dict_from_url(self.url, *args, **kwargs)

    WeightsEnum.get_state_dict = get_state_dict  # type: ignore[assignment]


def _create_backbone(model_name: str, dataset: str, num_classes: int) -> nn.Module:
    if "conv" in model_name:
        if dataset in ["cifar10", "cifar100"]:
            size = 32
        elif dataset == "tinyimagenet":
            size = 64
        elif dataset in ["imagenet-nette", "imagenet-woof", "imagenet-100"]:
            size = 128
        else:
            size = 224

        model = ConvNet(
            num_classes=num_classes,
            net_norm="batch",
            net_act="relu",
            net_pooling="avgpooling",
            net_depth=int(model_name[-1]),
            net_width=128,
            channel=3,
            im_size=(size, size),
        )
    elif model_name == "resnet18_modified":
        model = thmodels.__dict__["resnet18"](pretrained=False)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        model.maxpool = nn.Identity()
    elif model_name == "resnet101_modified":
        model = thmodels.__dict__["resnet101"](pretrained=False)
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        model.maxpool = nn.Identity()
    else:
        model = thmodels.__dict__[model_name](pretrained=False)

    return model


def _prune_classifier(model: nn.Module, classes: Iterable[int]) -> nn.Module:
    """Prune the last classifier weights to only keep given classes."""
    cls_indices: List[int] = list(classes)
    if not cls_indices:
        return model

    try:
        model_named_parameters = [name for name, _ in model.named_parameters()]
        for name, param in model.named_parameters():
            if name in {model_named_parameters[-1], model_named_parameters[-2]}:
                with torch.no_grad():
                    param.data = param[cls_indices]
    except Exception:
        print("ERROR in changing the number of classes.")
    return model


def create_model(
    model_name: str,
    dataset: str,
    classes: Iterable[int],
    pretrained: bool,
    data_root: str = "./data",
) -> nn.Module:
    """Create a model backbone and optionally load pretrained weights."""

    resolved_model_name = _resolve_model_name_for_dataset(model_name, dataset)
    classes_list = list(classes)
    num_classes = len(classes_list) if classes_list else 1000
    model = _create_backbone(resolved_model_name, dataset, num_classes)
    model = _prune_classifier(model, classes_list)

    if pretrained:
        if dataset in {
            "imagenet-100",
            "imagenet-10",
            "imagenet-nette",
            "imagenet-woof",
            "tinyimagenet",
            "cifar10",
            "cifar100",
        }:
            from pathlib import Path
            from data.download import ensure_pretrained_model

            pth_path = ensure_pretrained_model(
                dataset, resolved_model_name, Path(data_root)
            )
            checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint)
            model.load_state_dict(state)
        elif dataset == "imagenet-1k":
            if resolved_model_name == "efficientNet-b0":
                _patch_weights_enum()
            model = thmodels.__dict__[resolved_model_name](pretrained=True)

    return _ensure_feature_api(model)


def build_teacher(cfg: ExperimentConfig) -> nn.Module:
    """Build teacher model according to experiment configuration."""

    model = create_model(
        model_name=cfg.arch_name,
        dataset=cfg.subset,
        classes=cfg.classes,
        pretrained=True,
        data_root=cfg.data_root,
    )
    return model


def build_student(cfg: ExperimentConfig) -> nn.Module:
    """Build student model according to experiment configuration."""

    model = create_model(
        model_name=cfg.stud_name,
        dataset=cfg.subset,
        classes=cfg.classes,
        pretrained=False,
        data_root=cfg.data_root,
    )
    return model
