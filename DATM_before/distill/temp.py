import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch

# print("CUDA_LAUNCH_BLOCKING set")
# print("Before creating tensor")
# a = torch.tensor([1.0, 2.0]).cuda()
# torch.cuda.synchronize()
# print("After creating tensor")


print("CUDA_LAUNCH_BLOCKING set")

# 你的 CUDA 代码
print("Before CUDA call")
# CUDA 相关代码，如张量运算、模型前向传播等
print("After CUDA call")

print(torch.cuda.is_available())  # 确认 CUDA 是否可用
print(torch.cuda.current_device())  # 当前使用的 CUDA 设备
print(torch.cuda.get_device_name(0))  # 获取设备名称