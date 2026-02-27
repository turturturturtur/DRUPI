# Refraction 重构说明

`refraction/` 是对原始 RDED 代码的模块化重构，实现了更清晰的分层结构与可维护的 API。本文件说明新结构、使用方法以及与旧代码的对应关系。

## 目录结构与模块职责

- `refraction/__init__.py`：包初始化与简单说明。
- `refraction/config.py`：
  - 定义 `ExperimentConfig` 数据类，统一管理实验配置（数据集、模型、训练、蒸馏、路径等）。
  - 提供 `build_arg_parser()` 与 `parse_args_to_config()`，从命令行参数构造完整配置，并在函数内完成所有默认值和派生逻辑（不在 import 阶段做副作用）。
- `refraction/augmentation.py`：
  - `MultiRandomCrop`：合成阶段的多次随机裁剪生成 patch。
  - `ShufflePatches`：重排图像 patch，用于增强蒸馏训练。
  - `normalize` / `denormalize`：统一的图像归一化与反归一化。
  - `mixup` / `cutmix` / `mix_aug`：封装 mixup / cutmix 增强。
- `refraction/data/`：
  - `datasets.py`：
    - `ImageFolder`：按照给定 `classes` 与 `ipc` 从 `root/000xx` 目录中抽样固定张数图片，支持内存缓存（`mem=True`）与 shuffle。
    - `build_raw_train_loader(cfg)`：读取真实训练集（如 ImageNet / CIFAR），为合成阶段提供 `DataLoader`。
    - `build_synth_train_loader(cfg)`：读取合成数据集（`cfg.syn_data_path`），为蒸馏训练提供 `DataLoader`。
    - `build_val_loader(cfg)`：读取真实验证集（`cfg.val_dir`），为评估 student 模型提供 `DataLoader`。
- `refraction/models/`：
  - `convnet.py`：实现通用 `ConvNet`，用于 `conv3/4/5/6` 等 backbone。
  - `factory.py`：
    - `_create_backbone()`：根据 `model_name` 与数据集类型创建基础网络结构（含 `resnet18_modified` 等特例）。
    - `_prune_classifier()`：按 `classes` 截断最后一层分类头的权重。
    - `create_model(model_name, dataset, classes, pretrained)`：统一构建模型并在需要时加载预训练权重（来自 `./data/pretrain_models/{dataset}_{model_name}.pth` 或 torchvision 官方权重）。
    - `build_teacher(cfg)` / `build_student(cfg)`：基于 `ExperimentConfig` 构造 teacher / student 模型。
- `refraction/synthesis/`：
  - `pipeline.py`：
    - 提供 `run_synthesis(cfg)`，完成 **真实数据 → 合成小数据集** 的全过程：
      1. 清理并创建 `cfg.syn_data_path` 目录。
      2. 使用 `build_teacher(cfg)` 加载并冻结 teacher 模型。
      3. 通过 `build_raw_train_loader(cfg)` 读取真实训练集。
      4. 使用内部 `_selector` 基于 teacher 的预测选择高信息 patch，并用 `_mix_images` 将 patch 拼回整图。
      5. 使用 `_save_images` 把合成图像以 `syn_data_path/class_id/*.jpg` 形式写盘。
- `refraction/distill/`：
  - `trainer.py`：
    - `DistillationTrainer(cfg)`：封装 KL 蒸馏训练逻辑：
      - 构造 teacher / student 模型（teacher 冻结，student 可训练）。
      - 基于 `build_synth_train_loader(cfg)` / `build_val_loader(cfg)` 构造训练与验证数据加载器。
      - 根据 `cfg.sgd` / `cfg.adamw_lr` 等配置构造优化器与 `LambdaLR` 学习率调度器。
      - 在训练过程中使用 `mix_aug` 与温度缩放的 KL 散度进行蒸馏。
    - `train_one_epoch(epoch)` / `evaluate(epoch)`：单轮训练与评估逻辑。
    - `fit()`：完整训练循环，返回 `TrainingResult(best_acc1, best_epoch)`。
    - `run_distillation(cfg)`：简化入口，一次性完成蒸馏训练与评估。
- `refraction/utils/`：
  - `metrics.py`：`AverageMeter`、`accuracy` 等通用统计工具。
  - `optim.py`：`get_parameters(model)`，返回带/不带 weight decay 的参数组。
- `refraction/cli.py`：
  - 提供新的命令行入口，包含三个子命令：
    - `synthesize`：仅运行合成阶段。
    - `distill`：仅运行蒸馏训练与评估阶段。
    - `full-run`：先合成数据再进行蒸馏训练，相当于原来的单次完整实验。

## 新旧接口对照

- 旧入口：
  - `python main.py --subset ... --arch-name ... --stud-name ...`
  - 内部依赖 `argument.py` 在 import 阶段解析参数并创建结果目录。
- 新入口（推荐）：
  - 使用 `refraction` 的 CLI：
    - 只合成：
      - `python -m cli synthesize --subset cifar10 --arch-name conv3 --stud-name conv3 --ipc 10 --mipc 300 --factor 1 --num-crop 5`
    - 只蒸馏（假设合成数据已存在）：
      - `python -m cli distill --subset cifar10 --arch-name conv3 --stud-name conv3 --ipc 10 --mipc 300 --factor 1`
    - 完整流程：
      - `python -m cli full-run --subset cifar10 --arch-name conv3 --stud-name conv3 --ipc 10 --mipc 300 --factor 1 --num-crop 5`
  - 参数含义与原 `argument.py` 一致，`parse_args_to_config()` 会自动：
    - 根据 `subset` 设定 `nclass`、`classes`、`input_size`、`val_ipc` 等。
    - 依据 `ipc`、`nclass` 等推导 `re_batch_size` 与 `workers`。
    - 构造 `exp_name` 并将 `cfg.syn_data_path` 重定位到 `./exp/<exp_name>/<syn_data_path>`。
    - 根据 `mix_type` 与 `stud_name` 设定温度与默认 `adamw_lr`。

> 注意：旧的 `scripts/*.sh` 目前仍调用 `python main.py ...`。如果需要，你可以将脚本中的命令替换为上述 `python -m cli full-run ...`，以直接使用新实现。

## 数据与预训练权重约定

- 数据集路径：
  - 仍然采用 `./data/<subset>/train/00000/*.jpg`、`./data/<subset>/val/00000/*.jpg` 的结构，详见 `README.md` 与 `prepare/*.md`。
  - `ExperimentConfig` 默认会把：
    - `train_dir` 设为 `./data/<subset>/train/`
    - `val_dir` 设为 `./data/<subset>/val/`
- 预训练权重路径：
  - 若 `subset` 属于 `{imagenet-100, imagenet-10, imagenet-nette, imagenet-woof, tinyimagenet, cifar10, cifar100}`，则：
    - `factory.create_model()` 会从 `./data/pretrain_models/{subset}_{model_name}.pth` 中加载字典（优先取 `"model"` key）。
  - 若 `subset == "imagenet-1k"`，则：
    - 默认使用 torchvision 提供的官方预训练权重（必要时对 `WeightsEnum` 做兼容性 patch）。

## Python API 使用示例

可以在不通过 CLI 的情况下，直接在 Python 中调用 Refraction：

```python
from config import ExperimentConfig, finalize_config
from synthesis import run_synthesis
from distill import run_distillation

cfg = ExperimentConfig(subset="cifar10", arch_name="conv3", stud_name="conv3", ipc=10, mipc=300, factor=1)
cfg = finalize_config(cfg)

run_synthesis(cfg)
result = run_distillation(cfg)
print(result.best_acc1, result.best_epoch)
```

## 对未来开发者的扩展建议

- **添加新数据集**：
  - 在 `config.apply_dataset_defaults()` 中补充新的 `subset` 分支，指定 `nclass`、`classes`、`input_size`、`val_ipc` 等。
  - 确保 `./data/<subset>/train/000xx` 与 `./data/<subset>/val/000xx` 的目录格式与现有一致。
- **添加新模型**：
  - 若是 Conv 类型变体，可在 `ConvNet` 基础上调整 `net_depth` / `net_width` 或添加新的构造函数。
  - 若是 torchvision 内置模型，在 `factory._create_backbone()` 中增加对应分支，并在需要时写好预训练权重文件名规则。
- **修改训练策略**：
  - 与“合成”相关的变更，集中改动 `refraction/synthesis/pipeline.py`。
  - 与“蒸馏/重训练”相关的变更，集中改动 `refraction/distill/trainer.py`（例如替换 KL 损失、修改调度策略、使用不同增强方式等）。

