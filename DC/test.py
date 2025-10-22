# load pretrained/ConvNet/CIFAR10_best_features_train.pth
import torch 


features = torch.load('pretrained/ConvNet/CIFAR10_best_features_train.pth')

print(features.size())


# average pooling on h and w 
feature_pool = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  

print(feature_pool.size())