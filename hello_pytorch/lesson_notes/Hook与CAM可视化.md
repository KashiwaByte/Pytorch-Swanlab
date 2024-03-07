# Hook 函数
pytorch是动态图，中间内容会被释放，Hook函数来保留需要的信息
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305211714.png)

## tensor_register_hook
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305211808.png)
捕获非叶子节点 a b
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305211834.png)


## register_forward_hook

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305212127.png)



## register_forward_pre_hook
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305212652.png)

## register_backward_hook
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305212712.png)


# CAM可视化
最后一个网络层的特征图加权值，获得注意力
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305213129.png)


## Grad-CAM
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305213331.png)
通过grad-cam帮助我们判断学到的特征是否针对有效
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305213546.png)
