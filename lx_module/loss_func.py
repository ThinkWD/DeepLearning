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


class loss_cross_entropy_custom(object):
    def __call__(self, y_hat, y):
        """损失函数: 交叉熵损失的自定义实现"""
        return -torch.log(y_hat[range(len(y_hat)), y])
