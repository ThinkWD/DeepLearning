import torch

###########################################################
#
#  优化器 自定义实现
#
###########################################################


def opt_sgd(params, lr, weight_decay=0):
    '''优化算法: 批量随机梯度下降 (weight_decay 也可以只为某一层的参数单独设置, 这里直接为全局设置)'''
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)


def opt_adam(params, lr, weight_decay=0):
    '''优化算法: 批量随机梯度下降 (weight_decay 也可以只为某一层的参数单独设置, 这里直接为全局设置)'''
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def l2_penalty(w):
    '''权重衰减: 使用 L2 均方范数作为柔性限制, 来缓解过拟合'''
    return torch.sum(w.pow(2)) / 2


class opt_sgd_custom(object):
    def __init__(self, params, lr, weight_decay=0):
        """优化算法: 批量随机梯度下降的自定义实现"""
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay

    def __call__(self, batch_size):
        # 因为计算损失时，是 batch_size 个数据一起计算的
        # 所以更新参数时，需要将 batch_size 个数据的梯度除以 batch_size
        with torch.no_grad():  # 这里设置更新的时候不要参与梯度计算
            for param in self.params:  # 遍历所有参数, 可能是权重可能是偏差
                # 显式应用权重衰减: 计算损失时添加正则化项(需要计算L2范数), 然后基于这个损失计算梯度和更新参数
                # 隐式应用权重衰减: 更新参数时对参数直接添加惩罚项(不用计算L2范数), 可以看作是将正则化分散到每次更新中
                # 两种方式在实现上有所不同, 但在数学上、效果上是等价的。
                param -= self.lr * (param.grad / batch_size + self.weight_decay * param)  # 更新参数
                param.grad.zero_()  # 手动将梯度设置为零, 避免累积
