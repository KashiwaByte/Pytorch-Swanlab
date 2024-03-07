
迁移学习
源域的知识应用到目标域
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306185112.png)

通常认为特征提取的部分是有共性的，分类器的部分是需要改变的
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306185249.png)
微调的基本步骤
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240306185557.png)


微调方法
冻结卷积层
```
  

# 法1 : 冻结卷积层

# flag_m1 = 0

flag_m1 = 1

if flag_m1:

    for param in resnet18_ft.parameters():

        param.requires_grad = False

    print("conv1.weights[0, 0, ...]:\n {}".format(resnet18_ft.conv1.weight[0, 0, ...]))
```
卷积层学习率调小
```
# 法2 : conv 小学习率

# flag = 0

flag = 1

if flag:

    fc_params_id = list(map(id, resnet18_ft.fc.parameters()))     # 返回的是parameters的 内存地址

    base_params = filter(lambda p: id(p) not in fc_params_id, resnet18_ft.parameters())

    optimizer = optim.SGD([

        {'params': base_params, 'lr': LR*0},   # 0

        {'params': resnet18_ft.fc.parameters(), 'lr': LR}], momentum=0.9)

  

else:

    optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)              
```