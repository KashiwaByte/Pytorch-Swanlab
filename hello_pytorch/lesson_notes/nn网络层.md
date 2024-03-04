# 卷积层
特征检测器
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303162914.png)

## 卷积核的维度
卷积核在几个维度滑动就是几维的
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303163229.png)

## nn.Conv2d
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303163328.png)
padding保持输入输出图像大小不变
dilation提升感受野，有些部分是空出的

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303163509.png)

尺寸变化
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303164500.png)



## 转置卷积
输入尺寸小，输出尺寸大
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303165947.png)

卷积核和形状与正常卷积刚好是转置关系但是权值不同不是可逆关系
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303170316.png)
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303170418.png)


# 池化层
总结，最大值和平均值
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303170940.png)

## nn.MaxPool2d
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303171059.png)

反池化
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303172216.png)

## Linear
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303173955.png)


## 激活函数层
提供非线性
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303174747.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303174848.png)
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303174906.png)

Leaky 负半轴小斜率  蓝色
P 可学  橙色
R 振动 绿色
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303174952.png)


