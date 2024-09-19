import torch
from matplotlib import pyplot as plt


def accuracy(y_hat, y):
    """准确率, 计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 沿列方向找出每一行中最大概率对应的下标
    cmp = y_hat.type(y.dtype) == y  # 与真实值比较
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的数量


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的准确率"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式 评估模式不会执行计算梯度的操作, 性能会好一点
    num_samples = 0  # 样本总数
    num_accuracy = 0  # 预测正确的样本数
    with torch.no_grad():
        for X, y in data_iter:
            num_samples += y.numel()  # 这个批次的样本总数
            num_accuracy += accuracy(net(X), y)  # 这个批次预测正确的数量
    return num_accuracy / num_samples  # 分类正确的样本数 / 总样本数 = 精度


def train_epoch(net, opt, loss, train_iter):
    """训练模型一个迭代周期"""
    loss_sum = 0  # 训练损失
    num_samples = 0  # 样本总数
    num_accuracy = 0  # 预测正确的样本数
    if isinstance(net, torch.nn.Module):
        net.train()  # 如果是 nn 模型, 将模型设置为训练模式
    for X, y in train_iter:
        y_hat = net(X)  # 计算梯度并更新参数
        l = loss(y_hat, y)
        if isinstance(opt, torch.optim.Optimizer):  # 使用PyTorch内置的优化器和损失函数
            opt.zero_grad()
            l.mean().backward()
            opt.step()
            loss_sum += float(l) * len(y)
            num_samples += y.size().numel()
        else:  # 使用定制的优化器和损失函数
            l.sum().backward()
            opt(X.shape[0])
            loss_sum += float(l.sum())
            num_samples += y.numel()
        num_accuracy += accuracy(y_hat, y)
    # 返回训练损失和训练精度
    return loss_sum / num_samples, num_accuracy / num_samples


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale='linear',
        yscale='linear',
        fmts=('-', 'm--', 'g-.', 'r:'),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

    def save(self, filename):
        self.fig.savefig(filename)
