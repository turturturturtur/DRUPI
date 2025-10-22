import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
import numpy as np
from PIL import Image
from utils import get_dataset, get_network, get_all_features
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--class_num', default=10, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--data_path', type=str, default='~/data', help='dataset path')
parser.add_argument('--model', type=str, default='ConvNet', help='model')



args = parser.parse_args()
print(args)

BATCH_SIZE = 128
LR = 0.01

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

# make dst_train normalized
dst_train.transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainloader = torch.utils.data.DataLoader(
    dst_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

net = get_network(args.model, channel, num_classes, im_size).to(device) # get a random modelnet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)





if not os.path.exists(f"pretrained/{args.model}"):
    os.makedirs(f"pretrained/{args.model}")

if __name__ == "__main__":
    best_acc = 0
    best_feature = None
    print("Start Training")
    train_loss, train_acc, test_acc = [], [], []
    for epoch in tqdm(range(args.epoch), desc='Epoch', ncols=100, mininterval=1):
        if epoch in [args.epoch//3, args.epoch//3 * 2, args.epoch - 5]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            sum_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            # if i % 20 == 0:
            #     print('[epoch:%d, iter:%d] Loss: %.03f, %.03f  Acc: %.4f%%'
            #           % (epoch + 1, (i + 1 + epoch * length), sum_loss / total, loss.item(),
            #              100 * correct / total))
        train_loss.append(sum_loss / total)
        train_acc.append(100 * correct / total)

        acc1 = 0
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            acc1 = (100 * correct / total).item()
            if acc1 > best_acc:
                best_acc = acc1
                # save best_features
                best_features_train = get_all_features(net, trainloader, device)
                torch.save(best_features_train, f"pretrained/{args.model}/{args.dataset}_best_features_train.pth")
                best_features_test = get_all_features(net, testloader, device)
                torch.save(best_features_test, f"pretrained/{args.model}/{args.dataset}_best_features_test.pth")
                # save the best checkpoints
                torch.save(net.state_dict(), f"pretrained/{args.model}/{args.dataset}_best_model.pth")
                print('Best Test Set Accuracy: %.4f%% at epoch %d' % (acc1, epoch))
            test_acc.append(acc1)
        if epoch % 10  == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.plot(train_loss, label='train_loss')
            plt.legend()
            plt.subplot(122)
            plt.plot(train_acc, label='train_acc')
            plt.plot(test_acc, label='test_acc')
            plt.legend()
            plt.savefig(f"pretrained/{args.model}/{args.dataset}_train_test.png")
        
    print("Training Finished, TotalEPOCH=%d" % args.epoch)
    print ("Highest Accuracy is ", best_acc)

