# 简介
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305165325.png)

运行机制从脚本到硬盘到终端
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305165426.png)

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305170421.png)

# add scalar标量
创建runs的地址
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305170830.png)

记录标量
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305171150.png)

统计直方图
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305171406.png)


在图像处理和数据可视化中，"直方图"是一种统计图表，它展示了数据的分布情况。直方图通过将数据分组到连续的、不重叠的区间或"桶"中，并计算每个桶中的数据点数量来工作。在直方图中，"OVERLAY"和"OFFSET"是两个可用于调整直方图显示方式的概念。

OVERLAY： 直方图的"OVERLAY"是指在同一坐标轴上显示两个或多个直方图的方式。这种方法通常用于比较不同数据集的分布情况。在OVERLAY模式下，不同的直方图会使用不同的颜色或图案来区分，并且它们会重叠在一起，从而可以直观地比较它们的形状和范围。OVERLAY模式对于突出显示不同数据集之间的差异非常有用。

OFFSET： 直方图的"OFFSET"是指将直方图沿着X轴或Y轴移动一定距离的做法。OFFSET可以帮助区分重叠的直方图，使得每个直方图都能清晰地被观察到。在多个直方图重叠时，OFFSET可以通过沿X轴或Y轴移动某些直方图来改善可视化效果，减少视觉上的混乱。

总的来说，OVERLAY用于在同一视图中展示多个数据集的分布情况，而OFFSET用于调整直方图的位置，以改善多个直方图的可视化效果。在实际应用中，可以根据需要结合使用这两种方法来提高数据的可读性和比较的清晰度。

OFFSET
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305171926.png)

OVERLAY
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305171943.png)

# add_image
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305200416.png)

## grid
方便审查训练数据
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305201021.png)


## add_graph
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305205512.png)


## torch_summary
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305210127.png)
