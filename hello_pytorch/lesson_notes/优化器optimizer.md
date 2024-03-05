# 什么是优化器
管理并更新可学习参数，来降低loss
导数：函数在指定坐标轴的变化率
方向导数：指定方向上的变化率
梯度：一个向量，方向为方向导数取得最大值的方向，模长就是值

# 优化器基本属性
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303214155.png)


# 优化器基本方法
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303214323.png)


![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240303214536.png)
state_dict 和 load_state_dict 用于断点的时候恢复训练


## Momentum 动量
结合当前梯度与上一次更新信息，用于当前更新，通常设置为0.9
权重会变得指数下降
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305151830.png)
以下是改良后的参数更新公式
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305152211.png)


# torch.optism.SGD
SGD是最常用的优化器
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305152454.png)

Adam也是非常不错且常用的
# Pytorch的十类优化器
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305152547.png)
以下是相关论文
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240305152658.png)

SGD（随机梯度下降）： SGD是最基本的优化器，它的原理是在每次迭代中随机选择一个样本来计算梯度，并利用该梯度对参数进行更新。具体来说，对于参数θ，其更新公式为：θ = θ - lr * ∇θ，其中lr是学习率，∇θ是参数θ的梯度。SGD适用于大规模数据集和简单模型的训练，但在面对复杂模型和非凸优化问题时可能会陷入局部最优解。

Adagrad（自适应梯度算法）： Adagrad的原理是对每个参数自适应调整学习率，使得经常更新的参数具有较小的学习率，而不经常更新的参数具有较大的学习率。其更新公式为：θ = θ - lr / (sqrt(G) + ε) * ∇θ，其中G是参数θ的梯度平方的累积和，ε是一个很小的常数，用于防止除以零。Adagrad适用于稀疏数据和不同参数更新频率差异较大的情况。

Adadelta（自适应学习率方法）： Adadelta是在Adagrad的基础上改进的优化器，它解决了Adagrad学习率单调递减的问题。Adadelta不需要设置初始学习率，而是通过计算梯度的滑动平均值来自适应调整学习率。其更新公式为：θ = θ - (sqrt(Δθ + ε) / sqrt(G + ε)) * ∇θ，其中Δθ是参数θ的更新量的滑动平均值，G是参数θ的梯度平方的滑动平均值。Adadelta适用于需要精细调整学习率的复杂模型训练。

Adam（自适应矩估计）： Adam优化器结合了Momentum和RMSprop的优点。它计算梯度的一阶矩估计（均值）和二阶矩估计（未中心化的方差），并使用这些估计来调整每个参数的学习率。其更新公式为：θ = θ - lr * (m_t / (sqrt(v_t) + ε))，其中m_t和v_t分别是梯度的一阶和二阶矩估计。Adam是一种非常流行的优化器，适用于大多数深度学习任务。

Adamax： Adamax是Adam的一个变体，其区别在于它使用无穷范数来计算参数的缩放比例，而不是二阶矩估计。这使得Adamax在某些情况下比Adam更稳定。但在实践中，Adam通常表现得更好。

SparseAdam： SparseAdam是专门为稀疏张量设计的优化器。它是对Adam优化器的修改，可以更有效地处理稀疏梯度。SparseAdam适用于处理包含大量零梯度的数据集。

ASGD（平均随机梯度下降）： ASGD是SGD的一种变体，它维护一个参数的移动平均值，并在训练过程中使用这个平均值进行更新。这可以帮助减少参数更新的方差，并可能导致更稳定的收敛。ASGD适用于大规模数据集和在线学习任务。

Rprop（弹性反向传播）： Rprop是一种只基于梯度符号的优化器，它为每个参数维护一个单独的学习率。Rprop在处理某些非凸优化问题时效果很好，但它不适用于小批量或在线学习。

LBFGS（限制内存的BFGS）： LBFGS是BFGS（拟牛顿法）的一种变体，它使用有限的内存来逼近海森矩阵（二阶导数矩阵），从而减少计算和存储成本。LBFGS适用于小规模数据集和需要精确优化的任务。

RMSprop（均方根优化）： RMSprop是一种自适应学习率的优化器，它通过除以梯度的滑动平均平方根来调整学习率。这使得RMSprop在训练深度神经网络时非常有效，特别是在面对非平稳目标和RNNs时。