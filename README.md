以下是基于[Pytorch框架班](https://wx32e0ad0076a9091c.h5.xiaoe-tech.com/v1/course/video/v_5e9e5f6ddcef2_TCLvUDOF?type=2&pro_id=p_5df0ad9a09d37_qYqVmt85) 课程的Swanlab实践指南。

## Swanlab实践
lesson-08:[transforms_methods](hello_pytorch/lesson/lesson-08/transforms_methods_1.py) 介绍了Swanlab对预处理图像的记录  
lesson-17:[create_optimizer](hello_pytorch/lesson/lesson-17/create_optimizer.py) 介绍了Swanlab对基础二分类模型的全流程记录   
lesson-23:[hook_for_grad_cam](hello_pytorch/lesson/lesson-23/hook_for_grad_cam.py) 介绍了Swanlab对hook函数处理获得图像的记录  
lesson-24:[L2_regularization](hello_pytorch/lesson/lesson-24/L2_regularization.py) 介绍了Swanlab对L2正则化处理模型训练的记录，包含了与Tensorboard的对比  
lesson-25:[dropout_regularization](hello_pytorch/lesson/lesson-25/dropout_regularization.py) 介绍了Swanlab对dropout正则化处理模型训练的记录，包含了与Tensorboard的对比  
lesson-29:[finetune_resnet18](hello_pytorch/lesson/lesson-29/finetune_resnet18.py) 介绍了Swanlab的多实验对比，包括不添加finetune，只添加finetune，添加finetune并冻结卷积层，添加finetune并调小卷积层学习度等四类情况    
lesson-32:[resnet_inference](hello_pytorch/lesson/lesson-32/resnet_inference.py) 介绍了Swanlab对模型推理的记录，包含推理时间和推理效果  
lesson-33:[unet_portrait_matting](hello_pytorch/lesson/lesson-33/2_unet_portrait_matting.py) 介绍了Swanlab对图像分割模型U-net的加载与训练，包含提取效果图和原图的对比记录






## 课程安排及资料下载

### 🍬[作业讲解代码下载地址](https://github.com/greebear/pytorch-learning)

### 🍬[Week 1](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week1.md)
1. Pytorch简介及环境配置
2. Pytorch基础数据结构——张量
3. 张量操作与线性回归
4. 计算图与动态图机制
5. autograd与逻辑回归

### 🍚[Week2](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week2.md)
1. 数据读取机制DataLoader与Dataset
2. 数据预处理transforms模块机制
3. 二十二种transforms数据预处理方法
4. 学会自定义transforms方法

### 🍜[Week3](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week3.md)
1. nn.Module与网络模型构建步骤
2. 模型容器与AlexNet构建
3. 网络层中的卷积层
4. 网络层中的池化层、全连接层和激活函数层

### 🍖[Week4](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week4.md)
1. 权值初始化
2. 损失函数（一）
3. Pytorch的14种损失函数
4. 优化器optimizer的概念
5. torch.optim.SGD

### 🍹[Week5](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week5.md)
1. 学习率调整
2. TensorBoard简介与安装
3. TensorBoard使用（一）
4. TensorBoard使用（二）
5. hook函数与CAM

### 🍦[Week6](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week6.md)
1. weight_decay
2. dropout
3. Batch Normalization
4. Layer Normalization、Instance
5. Normalization和Group Normalization

### 🍭[Week7](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week7.md)
1. 模型保存与加载
2. Finetune
3. GPU的使用
4. Pytorch中常见报错

### 🍷[Week8](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week8.md)
1. 图像分类一瞥
2. 图像分割一瞥
3. 目标检测一瞥（上）
4. 目标检测一瞥（下）

### 🍾[Week9](https://github.com/JansonYuan/Pytorch-Camp/blob/master/Week9.md)
1. 对抗生成网络一瞥
2. 循环神经网络一瞥
