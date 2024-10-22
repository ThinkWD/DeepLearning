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


class Conv2D(torch.nn.Module):
    '''二维卷积层'''

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
