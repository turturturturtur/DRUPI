# Background

对于任何已有的Matching-based的DD方法，基本都是这样的框架：

- 初始化一个合成数据集 $D_{syn}=\{(x_i,y_i)\}$ ,
- 使得一个模型 $f=E\cdot C$ 在合成数据集上进行训练以最小化损失 $\ell =CE(f(x),y)$ 以更新模型，
- 通过Matching loss去对 $D_{syn}$ 进行优化。
- 这里$E$表示Feature Encoder， $C$ 表示Linear Classifier。
- 在数据集合成之后，我们在 $D_{syn}$ 上同样使用 $\ell =CE(f(x),y)$ 去训练模型来评估数据质量

# Idea

- 将合成数据集建模为 $D_{syn}=\{(x_i,y_i,\textcolor{red}{F_i})\}$
- 在合成数据集上进行训练以最小化损失 $\ell' =CE(f(x),y)+\textcolor{red}{\lambda \cdot MSE(F,E(x))}$
- 通过Matching Loss去对 $D_{syn}$ 进行优化。
- 这里 $F_i$ 表示对于 $i$-th 样本去优化的一个ground-truth feature。 $E(x_i)$ 表示对于 $x_i$ 图像，在合成数据集上训练的模型所提取到的特征。
- 在数据集合成之后，我们在 $D_{syn}$ 上使用 $\ell' =CE(f(x),y)+\textcolor{red}{\lambda \cdot MSE(F,E(x))}$ 去训练模型。

# Note

1. 跟Feature Matching的区别：Feature Matching本身优化的数据还是 $(x_i,y_i)$，而我们是 $(x_i,y_i,F_i)$。Feature Matching是在计算Matching Loss的时候引入了关于Feature的Loss，我们并没有修改这个Loss。
2. 如何理解 $F_i$ 的意义： $F_i$ 可以被建模为一张图像一个，这样表示这个图像对应的最佳特征值。也可以被建模为一个类的图像共用一个，这样可以表示这个类的图像的最佳特征Prototype。总之，它给在合成数据集上训练的模型提供了额外的监督信息，以提高合成数据集上训练的模型的表现。
3. $F_i$ 的其他选择： $F_i$ 不仅可以是 $R^{C\times H \times W}$ 的一个特征，也可以是对这个特征进行一定变换后的值，例如
   1. 按照C维度平均得到一个 $R^{H\times W}$ 的值，可以作为一种空间注意力的Label，
   2. 按照 $H\times W$ 维度avgpool得到一个 $R^{C}$ 的值，可以作为一种Channel注意力的Label

# Experiment

具体的流程，以DC为例
我们的 $D_{syn}$ 包括3个部分 $x_i,y_i,F_i$

- 每个Iteration
  - Evaluate $D_{syn}$ 质量
    - 在 $D_{syn}$ 上训练一批网络，使用的loss为 $\ell' =CE(f(x),y)+\textcolor{red}{\lambda \cdot MSE(F,E(x))}$
    - 在原始测试集上测试网络
  - 初始化网络、数据集
  - 合成数据集：每一个外部循环
    - 只学习$x$: match 原始数据集图片的CE梯度，合成数据集图片的CE梯度。只用这个matching loss。
    - 只学习$F$:
      - loss1 =  match 原始数据集的CE梯度，合成数据集图片的CE + $\lambda$ * 合成数据集特征的MSE 和的梯度
      - loss2 = $CE(C(F),y)$
      - 总loss = loss1 + $\lambda_2$ * loss2
    - 内部循环，更新网络
      - 在 $D_{syn}$ 上训练网络，使用的loss为 $\ell' =CE(f(x),y)+\textcolor{red}{\lambda \cdot MSE(F,E(x))}$

# Code

参数意思：

1. lbd: 控制feature 合成的MSE loss的系数，按照数据蒸馏里面的做法，一般建议为0.01-0.1，但是不一定
2. lr_feat 学习率
3. feat opt：优化器
4. pooling: 对feature pooling，比如 (bs, 4, 4) -> avgpool -> (bs, 1, 1)
5. layer idx，这里cifar10对应convnet 一共3层，为none就是最后一层。可以考虑不用最后一层的feature来合成
6. feat lbd：为了让feature 学习的更好加入的CE loss检测这部分feature喂进去网络做分类效果如何
7. n-feat：合成feature数量
8. use-feature：合成feature数量>1的时候，怎么利用多个feature？
   1. mean：用这些feature的平均值
   2. random: 随机用一个feature



# Run

## Install

```
cd DATM_FD
conda env create -f environment.yml

conda activate DATM-FD
```
## DC
### Install Environment

```
cd ~/DRUPI-main/DC
# use Python 3.13 to create environment
uv venv --python /usr/bin/python3.13
source .venv/bin/activate
uv pip install numpy scipy matplotlib tqdm
uv pip install torch torchvision
cd ~/DRUPI-main/DC/batch_invariant_ops
uv pip install -e .
cd ..
```
或从requirements.txt安装
### Run DC
```
python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR10' --init real --ipc 9 --lbd 0.1 --lbd-contrast 0.05 --batch_invariant 'eval' --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256 --generate_pretrained
python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR10' --init real --ipc 9 --lbd 0.1 --lbd-contrast 0.05 --batch_invariant 'eval' --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256

python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR10' --init real --ipc 48 --lbd 0.01 --lbd-contrast 0.05 --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256 --generate_pretrained
python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR10' --init real --ipc 48 --lbd 0.01 --lbd-contrast 0.05 --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256

python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR100' --init real --ipc 9 --lbd 0.01 --lbd-contrast 0.05 --batch_invariant 'eval' --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256 --generate_pretrained
python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR100' --init real --ipc 9 --lbd 0.01 --lbd-contrast 0.05 --batch_invariant 'eval' --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256

python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR100' --init real --ipc 48 --lbd 0.01 --lbd-contrast 0.05 --batch_invariant 'eval' --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256 --generate_pretrained
python DCmain3.0.py --data_path $填写你的data所在文件夹的绝对路径$ --dataset 'CIFAR100' --init real --ipc 48 --lbd 0.01 --lbd-contrast 0.05 --batch_invariant 'eval' --tta --tta_mode 'hflip'  --batch_real 256 --batch_train 256
```


## MTT

### ImageNet Subsets

#### buffer

```
cd MTT_FD
python buffer.py --dataset=ImageNet --subset=imagefruit --model=ConvNetD5 --train_epochs=50 --num_experts=100  --buffer_path=buffer_storage --data_path=/hpc2hdd/home/yxu409/wangshaobo/FD/MTT_FD/dataset/ImageNet/OpenDataLab___ImageNet-1K/raw/ImageNet-1K
python buffer.py --dataset=ImageNet --subset=imagenette --model=ConvNetD5 --train_epochs=50 --num_experts=100  --buffer_path=buffer_storage --data_path=/hpc2hdd/home/yxu409/wangshaobo/FD/MTT_FD/dataset/ImageNet/OpenDataLab___ImageNet-1K/raw/ImageNet-1K
python buffer.py --dataset=ImageNet --subset=imagewoof --model=ConvNetD5 --train_epochs=50 --num_experts=100  --buffer_path=buffer_storage --data_path=/hpc2hdd/home/yxu409/wangshaobo/FD/MTT_FD/dataset/ImageNet/OpenDataLab___ImageNet-1K/raw/ImageNet-1K
python buffer.py --dataset=ImageNet --subset=imagemeow --model=ConvNetD5 --train_epochs=50 --num_experts=100  --buffer_path=buffer_storage --data_path=/hpc2hdd/home/yxu409/wangshaobo/FD/MTT_FD/dataset/ImageNet/OpenDataLab___ImageNet-1K/raw/ImageNet-1K
python buffer.py --dataset=ImageNet --subset=imagesquawk --model=ConvNetD5 --train_epochs=50 --num_experts=100  --buffer_path=buffer_storage --data_path=/hpc2hdd/home/yxu409/wangshaobo/FD/MTT_FD/dataset/ImageNet/OpenDataLab___ImageNet-1K/raw/ImageNet-1K
python buffer.py --dataset=ImageNet --subset=imageyellow --model=ConvNetD5 --train_epochs=50 --num_experts=100  --buffer_path=buffer_storage --data_path=/hpc2hdd/home/yxu409/wangshaobo/FD/MTT_FD/dataset/ImageNet/OpenDataLab___ImageNet-1K/raw/ImageNet-1K
```
You can download MTTC100IPC48 buffer .pt files from [Huggingface](https://huggingface.co/likachan/MTT_BufferC100IPC48)
同时 data 文件夹中要有解压后的 cifar-100-python.tar.gz
```
cd MTT_FD
conda env create -n MTT -f environment.yml
conda activate MTT
# use at least four A100
CUDA_VISIBLE_DEVICES=4,5,6,7 python distill_pre.py --dataset=CIFAR100 --ipc=48 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --buffer_path ./buffers --data_path ../data --eval_it 100

CUDA_VISIBLE_DEVICES=4,5,6,7 python distill.py --dataset=CIFAR100 --ipc=48 --lbd 0.0005 --feat-lbd 0.0001 --pooling avg --zca --buffer_path ./buffers --data_path ../data --eval_it 100  --generate_pretrained

CUDA_VISIBLE_DEVICES=4,5,6,7 python distill_tta.py --dataset=CIFAR100 --ipc=48 --lbd 0.0005 --feat-lbd 0.0001 --pooling avg --zca --buffer_path ./buffers --data_path ../data --eval_it 100 --tta --tta_mode 'hflip' --Iteration 8000
```
### TinyImageNet
data 文件夹中要有解压后的 tiny-imagenet-200.tar.gz


## DATM

### TinyImageNet

### buffer

```
cd DATM_FD/buffer

python buffer_FTD.py --dataset=CIFAR100 --model=ConvNet --train_epochs=100 --num_experts=100 --zca --buffer_path=../buffer_storage/ --data_path=../dataset/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --lr_teacher=0.01 --mom=0. --batch_train=256

python buffer_FTD.py --dataset=Tiny --model=ConvNetD4 --train_epochs=100 --num_experts=100  --buffer_path=../buffer_storage/ --data_path=../dataset/ --rho_max=0.01 --rho_min=0.01 --alpha=0.3 --lr_teacher=0.01 --mom=0. --batch_train=256 --zca


```
