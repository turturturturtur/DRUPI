from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from config import ExperimentConfig, parse_args_to_config
from distill import run_distillation
from synthesis import run_synthesis


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Refraction CLI", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # synthesize
    synth_parser = subparsers.add_parser(
        "synthesize", help="run only the synthesis phase"
    )
    _attach_common_args(synth_parser)

    # distill
    distill_parser = subparsers.add_parser(
        "distill", help="run only the distillation phase"
    )
    _attach_common_args(distill_parser)

    # full run
    full_parser = subparsers.add_parser(
        "full-run", help="run synthesis followed by distillation"
    )
    _attach_common_args(full_parser)

    # prepare (dataset / model download)
    prepare_parser = subparsers.add_parser(
        "prepare", help="download and prepare datasets and/or pretrained models"
    )
    prepare_parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="dataset to prepare (cifar10, cifar100, tinyimagenet, etc.)",
    )
    prepare_parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="root directory for all datasets and pretrained models",
    )
    prepare_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="also download a pretrained model, e.g. 'conv3' or 'resnet18_modified'",
    )

    return parser


def _attach_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach arguments that map directly to ExperimentConfig."""

    # we simply reuse the parser from config and let it build the config
    from config import build_arg_parser

    base = build_arg_parser()
    for action in base._actions:
        if not action.option_strings:
            continue
        # avoid duplicating help option
        if action.option_strings[0] in ("-h", "--help"):
            continue
        parser._add_action(action)


def _build_config_from_namespace(ns: argparse.Namespace) -> ExperimentConfig:
    # Reuse parse_args_to_config by converting Namespace back to argv-like list
    arg_list = []
    for key, value in vars(ns).items():
        if key in {"command"}:
            continue
        if value is None or value is False:
            continue
        if key == "cos" and value is True:
            arg_list.append("--cos")
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                arg_list.append(flag)
        else:
            arg_list.extend([flag, str(value)])

    return parse_args_to_config(arg_list)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_main_parser()
    ns = parser.parse_args(list(argv) if argv is not None else None)

    if ns.command == "prepare":
        from data.download import ensure_dataset, ensure_pretrained_model

        data_root = Path(ns.data_root).resolve()
        ensure_dataset(ns.subset, data_root)
        if ns.model is not None:
            ensure_pretrained_model(ns.subset, ns.model, data_root)
        print("Preparation complete.")
        return

    cfg = _build_config_from_namespace(ns)

    # Auto-download dataset if needed (CIFAR, TinyImageNet)
    from data.download import ensure_dataset

    ensure_dataset(cfg.subset, Path(cfg.data_root).resolve())

    if ns.command == "synthesize":
        run_synthesis(cfg)
    elif ns.command == "distill":
        run_distillation(cfg)
    elif ns.command == "full-run":
        run_synthesis(cfg)
        run_distillation(cfg)
    else:
        parser.error(f"Unknown command {ns.command}")


if __name__ == "__main__":
    main()
