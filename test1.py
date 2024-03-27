import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
import swanlab 

path_lenet = os.path.abspath(os.path.join(BASE_DIR, "hello_pytorch",  "model", "lenet.py"))
path_tools = os.path.abspath(os.path.join(BASE_DIR, "hello_pytorch", "tools", "common_tools.py"))
sys.path.append(path_lenet)
sys.path.append(path_tools)
import sys
hello_pytorch_DIR = os.path.abspath(os.path.dirname(__file__)+os.path.sep+".."+os.path.sep+"..")
sys.path.append(hello_pytorch_DIR)

from hello_pytorch.model.lenet import LeNet
from hello_pytorch.tools.common_tools import transform_invert, set_seed

set_seed()  # 设置随机种子
rmb_label = {"1": 0, "100": 1}

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 4
LR = 0.001
log_interval = 10
val_interval = 1

# ============================ step 1/5 数据 ============================


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

swanlab.init(experiment_name="CIFAR10",config={"Epoch":10,"Batch_Size":4,"Learning_Rate":0.001,"log_iterval":10,"val_interval":1})

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomGrayscale(p=0.9),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = CIFAR10(train=True,transform=train_transform,download=True,root='./data')
valid_data = CIFAR10(train=False,download=True,transform=valid_transform,root='./data')

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================

class MyNet(nn.Module):

    def __init__(self,classes) -> None:
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features=800, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=classes)
        pass
        
      

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


    def initialize_weights(self):
        """使用Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


net = MyNet(classes=10)
net.initialize_weights()

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)                        # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()

for epoch in range(MAX_EPOCH):

    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()
    for i, data in enumerate(train_loader):

        # forward
        inputs, labels = data
        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
            swanlab.log({'Loss':loss_mean,'Accuracy':correct/total})
            loss_mean = 0.

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            valid_curve.append(loss_val)
            print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, correct_val / total_val))
            swanlab.log({"Loss_Val":loss_val,'Accuracy_Val':correct_val/total_val})


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
swanlab.log({"Result":swanlab.Image(plt)})
plt.show()

# # ============================ inference ============================

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# test_dir = os.path.join(BASE_DIR, "test_data")

# test_data = RMBDataset(data_dir=test_dir, transform=valid_transform)
# valid_loader = DataLoader(dataset=test_data, batch_size=1)

# for i, data in enumerate(valid_loader):
#     # forward
#     inputs, labels = data
#     outputs = net(inputs)
#     _, predicted = torch.max(outputs.data, 1)

#     rmb = 1 if predicted.numpy()[0] == 0 else 100

#     img_tensor = inputs[0, ...]  # C H W
#     img = transform_invert(img_tensor, valid_transform)
#     plt.imshow(img)
#     plt.title("LeNet got {} Yuan".format(rmb))
#     plt.show()
#     plt.pause(0.5)
#     plt.close()






