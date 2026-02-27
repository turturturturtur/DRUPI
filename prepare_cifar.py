from __future__ import annotations

import argparse
import os
from pathlib import Path

from tqdm import tqdm
from torchvision import datasets


def _save_split(
    subset: str,
    root: Path,
    split: str,
) -> None:
    """
    下载并按 RDED 约定格式保存一个 split（train 或 val/test）。

    目标目录结构：
        <root>/<subset>/<split>/
            00000/image00000.jpg
            00000/image00001.jpg
            ...
            00001/image00000.jpg
            ...
    """
    out_dir = root / subset / split
    os.makedirs(out_dir, exist_ok=True)

    is_train = split == "train"
    if subset == "cifar10":
        ds = datasets.CIFAR10(
            root=str(root / "torchvision"),
            train=is_train,
            download=True,
        )
    elif subset == "cifar100":
        ds = datasets.CIFAR100(
            root=str(root / "torchvision"),
            train=is_train,
            download=True,
        )
    else:
        raise ValueError(f"Unsupported subset: {subset}")

    for idx in tqdm(range(len(ds)), desc=f"{subset} {split}"):
        image, label = ds[idx]
        class_dir = out_dir / f"{label:05d}"
        os.makedirs(class_dir, exist_ok=True)
        image_path = class_dir / f"image{idx:05d}.jpg"
        image.save(image_path, format="JPEG")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        "Prepare CIFAR10/CIFAR100 in RDED folder format"
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["cifar10", "cifar100"],
        required=True,
        help="要下载的数据集名称",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="数据集根目录（默认 ./data）",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    subset: str = args.subset

    print(f"Preparing {subset} under {root}")

    _save_split(subset, root, "train")
    # RDED 中把验证集放在 val/ 下；CIFAR 的 test split 对应这里的 val
    _save_split(subset, root, "val")

    print("Done.")


if __name__ == "__main__":
    main()

