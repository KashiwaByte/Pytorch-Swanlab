
# Normalization介绍
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306145640.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306145801.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306151427.png)

# 批标准化介绍
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306113601.png)

## 批标准化算法 
求均值
求方差
标准化
affine transform 增强
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306113814.png)

bn层需要在卷积层之后，激活层之前
```
 def __init__(self, classes):

        super(LeNet_bn, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)

        self.bn1 = nn.BatchNorm2d(num_features=6)

  

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.bn2 = nn.BatchNorm2d(num_features=16)

  

        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        self.bn3 = nn.BatchNorm1d(num_features=120)

  

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, classes)

  

    def forward(self, x):

        out = self.conv1(x)

        out = self.bn1(out)

        out = F.relu(out)

  

        out = F.max_pool2d(out, 2)

  

        out = self.conv2(out)

        out = self.bn2(out)

        out = F.relu(out)
```


## BatchNorm基类
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306114534.png)


![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306114754.png)

## 3种类型的数据要求
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306115529.png)


批次 * 特征数 * 特征维度（1d 2d 3d)
batch_size     fnum_features   feature_shape

# Layer Normalization
无法直接采用BN
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306145858.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306150006.png)


# Instance Normalization
不同风格图像的特征不能混为一谈，需要逐通道计算均值和方差
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306150346.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306150502.png)


# Group Normalization
大模型，小batch样本，BN估计不准
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306150729.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306150921.png)


# 四类方法与差异
Normalization（标准化）方法是深度学习中常用的技术，用于稳定和加速神经网络的训练。以下是四种常见的Normalization方法及其差异：

1. Batch Normalization (BN)：
   - BN是一种在每个批次的数据上应用的标准化技术，它通过减少内部协变量偏移来加速训练。
   - 它独立地对每个特征通道进行标准化，计算每个批次中特征的均值和方差，并使用这些统计量将数据标准化。
   - BN在每个批次上进行操作，因此对批次大小比较敏感，较小的批次可能会影响其性能。

2. Layer Normalization (LN)：
   - LN是对单个样本中的所有特征进行标准化，它计算所有特征的均值和方差，并使用这些统计量进行标准化。
   - LN对批次大小不敏感，因此适用于批次大小为1或变化的情况，常见于循环神经网络（RNN）中。

3. Instance Normalization (IN)：
   - IN通常用于样式迁移任务中，它在单个样本的每个通道上独立进行标准化。
   - 它对每个样本的每个特征通道计算均值和方差，并使用这些统计量进行标准化。
   - IN忽略了批次中的其他样本，只关注单个实例，因此在样式迁移等任务中比较有效。

4. Group Normalization (GN)：
   - GN是BN的一种变体，它将特征通道分成小组，并在每组内进行标准化。
   - GN不依赖于批次大小，因此适用于批次大小受限或不一致的情况。
   - GN在每组内计算均值和方差，然后进行标准化，其性能通常不受批次大小影响。

总结：
- BN依赖于批次大小，适用于批次较大且稳定的情况。
- LN和IN对批次大小不敏感，LN在特征层面进行标准化，而IN在样本和通道层面进行标准化。
- GN是一种折衷方案，它在组内进行标准化，不受批次大小影响，适合各种大小的批次。

选择哪种Normalization方法取决于具体任务的需求、网络架构和训练数据的特点。