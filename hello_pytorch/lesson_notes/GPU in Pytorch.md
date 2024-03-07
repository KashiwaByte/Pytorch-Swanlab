
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306200957.png)

# 数据迁移
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306201008.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306201148.png)

tensor 要重新赋值
模型不需要重新赋值（自动执行inplace）

# cuda常用方法
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306201601.png)
逻辑GPU小于物理GPU
可以自己分配GPU，用于多任务或者多人协作的情况
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306201701.png)


# 多GPU分发并行
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306201849.png)
模型构建后，Parallel包装一下
```
 # model

    net = FooNet(neural_num=3, layers=3)

    net = nn.DataParallel(net)

    net.to(device)
```

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306202141.png)
