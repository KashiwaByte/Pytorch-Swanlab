# torchvision介绍
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302111732.png)

# transforms常用预处理方法
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302111814.png)

# Compose
有序组合和包装预处理方法
```
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
```

# Normalize处理
加快模型收敛速度
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302112440.png)


# 数据增强
## 总览
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302191511.png)

## 裁剪
中心裁剪
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302183459.png)

随机裁剪
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302183527.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302184138.png)


## 翻转 Flip
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302184604.png)

## 旋转 Rotation
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302184724.png)

## 填充Pad
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302185231.png)

## 色彩变换ColorJitter
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302185404.png)


## 灰度转换 Grayscale
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302185627.png)


## 仿射变换 Affine
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302185928.png)

##  遮挡 Erasing
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302190215.png)


## transforms操作
随机
概率
乱序

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302190806.png)


## 自定义transforms方法
接收一个参数，返回一个参数
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240302191054.png)

