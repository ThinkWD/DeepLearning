import torch

###########################################################
#
#  损失函数 自定义实现
#
###########################################################


def loss_squared():
    """损失函数: 平方误差, 又称 L2"""
    return torch.nn.MSELoss()


class loss_squared_custom(object):
    def __call__(self, y_hat, y):
        """损失函数: 平方误差, 又称 L2. 的自定义实现"""
        # 为了避免 y 与 y_hat 的形状不同, 对 y 做了一次 reshape
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def loss_cross_entropy():
    """损失函数: 交叉熵损失 (包含了 softmax 操作, 不需要在网络结构中重复定义)"""
    return torch.nn.CrossEntropyLoss()


def softmax(X):
    """softmax 函数
    1. 对每个项求幂 (使用exp)
    2. 对每一行求和得到每个样本的规范化常数
    3. 将每一行除以其规范化常数, 确保结果的和为 1
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # (广播机制)


class loss_cross_entropy_custom(object):
    def __call__(self, y_hat, y):
        """损失函数: 交叉熵损失的自定义实现 (包含了 softmax 操作, 不需要在网络结构中重复定义)"""
        y_hat = softmax(y_hat)
        return -torch.log(y_hat[range(len(y_hat)), y])
