import torch

###########################################################
#
#  优化器 自定义实现
#
###########################################################


def opt_sgd(params, lr):
    """优化算法: 批量随机梯度下降"""
    return torch.optim.SGD(params, lr=lr)


class opt_sgd_custom(object):
    def __init__(self, params, lr):
        """优化算法: 批量随机梯度下降的自定义实现"""
        self.params = params
        self.lr = lr

    def __call__(self, batch_size):
        # 因为计算损失时，是 batch_size 个数据一起计算的
        # 所以更新参数时，需要将 batch_size 个数据的梯度除以 batch_size
        with torch.no_grad():  # 这里设置更新的时候不要参与梯度计算
            for param in self.params:  # 遍历所有参数, 可能是权重可能是偏差
                param -= self.lr * param.grad / batch_size  # 更新参数
                param.grad.zero_()  # 手动将梯度设置为零, 避免累积
