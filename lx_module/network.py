import numpy
import torch


class BaseModel:
    def __init__(self):
        self.is_train = False

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


def dropout_layer(X, dropout, is_train=False):
    '''以 dropout 的概率随机丢弃输入 X 中的元素'''
    assert 0 <= dropout <= 1
    if dropout == 0 or is_train == False:
        return X
    if dropout == 1:
        return torch.zeros_like(X)
    # torch.rand 生成 0 ~ 1 之间的均匀随机分布, 大于 dropout 部分置1, 小于的部分置零, 得到 mask
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    # 在最后除以 1 - p 是为了保持输出的期望值不变。
    # 随机丢弃一部分神经元的输出会使得剩余的神经元输出变得稀疏。
    # 如果不进行调整，剩余神经元的输出总和会变小，从而影响模型的训练效果。
    return mask * X / (1.0 - dropout)


def relu(X):
    """激活函数 relu, 就是 max(x, 0)"""
    return torch.max(X, torch.zeros_like(X))


def corr2d(X, K):
    '''cross-correlation, 二维互相关运算. X 是输入, K 是卷积核. 两个都是 2D 矩阵'''
    k_h, k_w = K.shape
    Y = torch.zeros((X.shape[0] - k_h + 1, X.shape[1] - k_w + 1), device=X.device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + k_h, j : j + k_w] * K).sum()
    return Y


def corr2d_multi_in(X, Kernel):
    '''多通道的互相关运算. 先在每个通道上做二维互相关运算, 最后求和. 两个都是 3D 矩阵'''
    return sum(corr2d(x, k) for x, k in zip(X, Kernel))


def corr2d_multi_in_batch(X, K):
    '''支持批次维度的二维互相关运算. X 是 4D 张量, K 是 3D 张量'''
    batch_size, n_c, n_h, n_w = X.shape
    k_c, k_h, k_w = K.shape
    assert k_c == n_c
    Y = torch.zeros((batch_size, n_h - k_h + 1, n_w - k_w + 1), device=X.device)
    for b in range(batch_size):
        Y[b] = corr2d_multi_in(X[b], K)
    return Y


def corr2d_multi_in_out(X, Kernel):
    '''多输入+多输出的互相关运算. 使用 numpy.stack 将多个 2D 矩阵整合为 3D 矩阵'''
    assert len(X.shape) == 3 and len(Kernel.shape) == 4
    return numpy.stack([corr2d_multi_in(X, k) for k in Kernel], 0)


def pool2d(X, pool_size, mode='avg'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1), device=X.device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i : i + p_h, j : j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i : i + p_h, j : j + p_w].mean()
    return Y


class Conv2D(torch.nn.Module):
    '''二维卷积层'''

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def batch_normalization(X, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9):
    '''
    gamma 和 beta: 可学习的缩放和偏移
    eps: 一个很小的常数, 用于避免除 0 错误
    moving_mean 和 moving_var: 全局/整个数据集上的均值和方差, 用于预测模式
    momentum: [移动平均法]在不知道全局数据的情况下, 逐步推断平均值. 用来更新 moving_mean 和 moving_var
    '''
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用全局的均值和方差做归一化
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)  # 2 是全连接层, 4 是卷积层
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data


class BatchNorm(torch.nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = torch.nn.Parameter(torch.ones(shape))
        self.beta = torch.nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_normalization(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var
        )
        return Y
