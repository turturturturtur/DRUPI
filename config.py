from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single RDED / Refraction experiment."""

    # dataset / paths
    subset: str = "imagenet-1k"
    data_root: str = "./data"
    train_dir: str = "./data/imagenet-1k/train/"
    val_dir: str = "./data/imagenet-1k/val/"
    nclass: int = 1000
    classes: List[int] = field(default_factory=list)
    input_size: int = 224

    # synthesis
    arch_name: str = "resnet18"
    mipc: int = 600
    ipc: int = 50
    num_crop: int = 1
    factor: int = 2
    syn_data_path: str = "syn_data"

    # distillation / retrain
    stud_name: str = "resnet18"
    re_batch_size: int = 0
    re_accum_steps: int = 1
    re_epochs: int = 300
    workers: int = 4
    val_ipc: int = 30

    # augmentation
    mix_type: Optional[str] = "cutmix"
    mixup: float = 0.8
    cutmix: float = 1.0
    min_scale_crops: float = 0.08
    max_scale_crops: float = 1.0
    temperature: Optional[float] = None

    # optimization
    sgd: bool = False
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    adamw_lr: float = 0.0
    adamw_weight_decay: float = 0.01
    cos: bool = True

    # DRUPI privileged information
    use_feat_labels: bool = False
    lambda_reg: float = 0.5
    lambda_task: float = 0.1

    # bookkeeping
    seed: int = 42
    exp_name: Optional[str] = None
    exp_root: str = "./exp"

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


def apply_dataset_defaults(cfg: ExperimentConfig) -> None:
    """Apply dataset-specific defaults (classes, nclass, val_ipc, input_size)."""

    subset = cfg.subset

    if subset in {
        "imagenet-a",
        "imagenet-b",
        "imagenet-c",
        "imagenet-d",
        "imagenet-e",
        "imagenet-birds",
        "imagenet-fruits",
        "imagenet-cats",
        "imagenet-10",
    }:
        cfg.nclass = 10
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 50
        cfg.input_size = 224
    elif subset == "imagenet-nette":
        cfg.nclass = 10
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 50
        cfg.input_size = 224
        if cfg.arch_name in {"conv5", "conv6"} or cfg.stud_name in {"conv5", "conv6"}:
            cfg.input_size = 128
    elif subset == "imagenet-woof":
        cfg.nclass = 10
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 50
        cfg.input_size = 224
        if cfg.arch_name in {"conv5", "conv6"} or cfg.stud_name in {"conv5", "conv6"}:
            cfg.input_size = 128
    elif subset == "imagenet-100":
        cfg.nclass = 100
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 50
        cfg.input_size = 224
        if cfg.arch_name in {"conv5", "conv6"} or cfg.stud_name in {"conv5", "conv6"}:
            cfg.input_size = 128
    elif subset == "imagenet-1k":
        cfg.nclass = 1000
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 50
        cfg.input_size = 224
    elif subset == "cifar10":
        cfg.nclass = 10
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 1000
        cfg.input_size = 32
    elif subset == "cifar100":
        cfg.nclass = 100
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 100
        cfg.input_size = 32
    elif subset == "tinyimagenet":
        cfg.nclass = 200
        cfg.classes = list(range(cfg.nclass))
        cfg.val_ipc = 50
        cfg.input_size = 64

    # always keep nclass consistent with classes length if classes has been set
    if cfg.classes:
        cfg.nclass = len(cfg.classes)


def infer_training_hyperparams(cfg: ExperimentConfig) -> None:
    """Infer batch size, workers, experiment name and derived paths."""

    # train / val directories are derived from data_root and subset
    data_root = Path(cfg.data_root).resolve()
    cfg.train_dir = str(data_root / cfg.subset / "train")
    cfg.val_dir = str(data_root / cfg.subset / "val")

    if cfg.re_batch_size == 0:
        if cfg.ipc == 50:
            cfg.re_batch_size = 100
            cfg.workers = 4
        elif cfg.ipc == 10:
            cfg.re_batch_size = 50
            cfg.workers = 4
        elif cfg.ipc == 1:
            cfg.re_batch_size = 10
            cfg.workers = 0

        if cfg.nclass == 10:
            cfg.re_batch_size *= 1
        if cfg.nclass == 100:
            cfg.re_batch_size *= 2
        if cfg.nclass == 1000:
            cfg.re_batch_size *= 2

        if cfg.subset == "tinyimagenet":
            cfg.re_batch_size = 100

    # reset batch size below ipc * nclass
    max_batch = cfg.ipc * cfg.nclass
    if cfg.re_batch_size > max_batch:
        cfg.re_batch_size = int(max_batch)

    # reset batch size with gradient accumulation
    if cfg.re_accum_steps != 1:
        cfg.re_batch_size = int(cfg.re_batch_size / cfg.re_accum_steps)

    # experiment naming and synthetic data root (no directory creation here)
    if cfg.exp_name is None:
        cfg.exp_name = f"{cfg.subset}_{cfg.arch_name}_f{cfg.factor}_mipc{cfg.mipc}_ipc{cfg.ipc}_cr{cfg.num_crop}"

    exp_dir = Path(cfg.exp_root).resolve() / cfg.exp_name
    cfg.syn_data_path = str(exp_dir / cfg.syn_data_path)


def infer_optimizer_hyperparams(cfg: ExperimentConfig) -> None:
    """Infer temperature and optimizer defaults based on mix type and model."""

    # temperature for distillation
    if cfg.mix_type == "mixup":
        cfg.temperature = 4
    elif cfg.mix_type == "cutmix":
        cfg.temperature = 20

    # AdamW learning rate defaults based on student architecture
    stud = cfg.stud_name
    if stud == "vgg11":
        cfg.adamw_lr = 0.0005
    elif stud in {"conv3", "conv4", "conv5", "conv6"}:
        cfg.adamw_lr = 0.001
    elif stud in {"resnet18", "resnet18_modified", "resnet50", "resnet101", "resnet101_modified"}:
        cfg.adamw_lr = 0.001
    elif stud == "efficientnet_b0":
        cfg.adamw_lr = 0.002
    elif stud == "mobilenet_v2":
        cfg.adamw_lr = 0.0025
    elif stud == "alexnet":
        cfg.adamw_lr = 0.0001
    elif stud in {"vit_b_16", "swin_v2_t"}:
        cfg.adamw_lr = 0.0001

    # special experiment setting
    if cfg.subset == "cifar100" and cfg.arch_name == "conv3" and cfg.stud_name == "conv3":
        cfg.re_batch_size = 25
        cfg.adamw_lr = 0.002


def finalize_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Run all post-processing steps to obtain a fully specified config."""

    apply_dataset_defaults(cfg)
    infer_training_hyperparams(cfg)
    infer_optimizer_hyperparams(cfg)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    """Create an ArgumentParser compatible with the original interface."""

    parser = argparse.ArgumentParser("Refraction / RDED")

    # synthesis
    parser.add_argument("--arch-name", type=str, default="resnet18")
    parser.add_argument("--subset", type=str, default="imagenet-1k")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--nclass", type=int, default=None)
    parser.add_argument("--mipc", type=int, default=600)
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--num-crop", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--factor", type=int, default=2)

    # retrain
    parser.add_argument("--re-batch-size", type=int, default=0)
    parser.add_argument("--re-accum-steps", type=int, default=1)
    parser.add_argument(
        "--mix-type",
        default="cutmix",
        type=str,
        choices=["mixup", "cutmix", None],
    )
    parser.add_argument("--stud-name", type=str, default="resnet18")
    parser.add_argument("--val-ipc", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--classes", type=str, default=None, help="comma separated class ids")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--min-scale-crops", type=float, default=0.08)
    parser.add_argument("--max-scale-crops", type=float, default=1.0)
    parser.add_argument("--re-epochs", type=int, default=300)
    parser.add_argument("--syn-data-path", type=str, default="syn_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--cos", action="store_true", default=True)
    parser.add_argument("--no-cos", dest="cos", action="store_false")

    # optimization
    parser.add_argument("--sgd", action="store_true", default=False)
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.1,
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adamw-lr", type=float, default=0.0)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.01)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exp-root", type=str, default="./exp")

    # DRUPI privileged information
    parser.add_argument("--use-feat-labels", action="store_true", default=False)
    parser.add_argument("--lambda-reg", type=float, default=0.5)
    parser.add_argument("--lambda-task", type=float, default=0.1)

    return parser


def _parse_classes_arg(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    parts = [p for p in raw.split(",") if p]
    return [int(p) for p in parts]


def parse_args_to_config(argv: Optional[Iterable[str]] = None) -> ExperimentConfig:
    """Parse command line arguments and return a fully initialized config."""

    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = ExperimentConfig()
    cfg.subset = args.subset
    cfg.data_root = args.data_root
    cfg.arch_name = args.arch_name
    cfg.mipc = args.mipc
    cfg.ipc = args.ipc
    cfg.num_crop = args.num_crop
    cfg.factor = args.factor
    cfg.stud_name = args.stud_name
    cfg.re_batch_size = args.re_batch_size
    cfg.re_accum_steps = args.re_accum_steps
    cfg.mix_type = args.mix_type
    cfg.workers = args.workers
    cfg.min_scale_crops = args.min_scale_crops
    cfg.max_scale_crops = args.max_scale_crops
    cfg.re_epochs = args.re_epochs
    cfg.syn_data_path = args.syn_data_path
    cfg.seed = args.seed
    cfg.mixup = args.mixup
    cfg.cutmix = args.cutmix
    cfg.cos = args.cos
    cfg.sgd = args.sgd
    cfg.learning_rate = args.learning_rate
    cfg.momentum = args.momentum
    cfg.weight_decay = args.weight_decay
    cfg.adamw_lr = args.adamw_lr
    cfg.adamw_weight_decay = args.adamw_weight_decay
    cfg.exp_name = args.exp_name
    cfg.exp_root = args.exp_root
    cfg.use_feat_labels = args.use_feat_labels
    cfg.lambda_reg = args.lambda_reg
    cfg.lambda_task = args.lambda_task

    # optional overrides
    if args.train_dir is not None:
        cfg.train_dir = args.train_dir
    if args.val_dir is not None:
        cfg.val_dir = args.val_dir
    if args.input_size is not None:
        cfg.input_size = args.input_size
    if args.val_ipc is not None:
        cfg.val_ipc = args.val_ipc
    if args.nclass is not None:
        cfg.nclass = args.nclass
    if args.classes is not None:
        parsed_classes = _parse_classes_arg(args.classes)
        if parsed_classes is not None:
            cfg.classes = parsed_classes
    if args.temperature is not None:
        cfg.temperature = args.temperature

    return finalize_config(cfg)

