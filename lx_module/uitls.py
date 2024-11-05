import time
import torch
from matplotlib import pyplot as plt


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


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


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def reset(self):
        self.times = []


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


def accuracy(y_hat, y):
    """准确率, 计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 沿列方向找出每一行中最大概率对应的下标
    cmp = y_hat.type(y.dtype) == y  # 与真实值比较
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的数量


def log_rmse(net, loss, data_iter):
    """计算在指定数据集上模型的相对损失 (对于回归任务, 结果数量级差距较大时)"""
    num_loss = 0  # 相对损失
    num_samples = 0  # 样本总数
    net.eval()  # 将模型设置为评估模式: 不计算梯度, 跳过丢弃法, 性能更好
    device = try_gpu()
    for X, y in data_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = torch.clamp(net(X), 1, float('inf'))
        l = loss(torch.log(y_hat), torch.log(y))
        if isinstance(loss, torch.nn.Module):
            num_loss += l * len(y)
            num_samples += y.size().numel()
        else:
            num_loss += l.sum()
            num_samples += y.numel()
    return float(torch.sqrt(num_loss / num_samples))  # 返回相对损失


def evaluate(net, data_iter, device=try_gpu()):
    """计算在指定数据集上模型的损失和精度"""
    num_samples = 0  # 样本总数
    num_accuracy = 0  # 预测正确的样本数
    net.eval()  # 将模型设置为评估模式: 不计算梯度, 跳过丢弃法, 性能更好
    with torch.no_grad():
        for X, y in data_iter:
            X = [x.to(device) for x in X] if isinstance(X, list) else X.to(device)
            y = y.to(device)
            num_samples += y.numel()
            num_accuracy += accuracy(net(X), y)
    return num_accuracy / num_samples


def train_batch(net, opt, loss, X, y, device=try_gpu()):
    X = [x.to(device) for x in X] if isinstance(X, list) else X.to(device)
    y = y.to(device)
    net.train()  # 将模型设置为训练模式: 更新参数, 应用丢弃法
    y_hat = net(X)
    l = loss(y_hat, y)  # 这个批次的损失
    if isinstance(opt, torch.optim.Optimizer):  # 计算梯度
        opt.zero_grad()  # 清空上次的梯度
        l.mean().backward()  # 根据损失函数计算梯度
        opt.step()  # 根据梯度更新参数
        num_loss = float(l) * len(y)  # 更新总损失
    else:
        l.sum().backward()  # 根据损失函数计算梯度
        opt(X.shape[0])  # 根据梯度更新参数
        num_loss = float(l.sum())  # 更新总损失
    num_accuracy = accuracy(y_hat, y)  # 这个批次预测正确的数量
    return num_loss, num_accuracy


def train_classification(net, opt, loss, data, num_epochs, log="log"):
    device = try_gpu()
    train_iter, test_iter = data.get_iter()
    timer = Timer()
    num_batches = len(train_iter)
    animator = Animator(ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    for ep in range(1, num_epochs + 1):
        sum_loss = 0
        sum_train_acc = 0
        sum_examples = 0
        sum_predictions = 0
        timer.reset()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(net, opt, loss, X, y, device)
            sum_loss += l
            sum_train_acc += acc
            sum_examples += y.shape[0]
            sum_predictions += y.numel()
            timer.stop()
            if (i + 1) % 50 == 0 or i == num_batches - 1:
                train_loss = sum_loss / sum_examples
                train_acc = sum_train_acc / sum_predictions
                animator.add((ep - 1) + (i + 1) / num_batches, (train_loss, train_acc, None))
                print(
                    f"[{log}] epoch {ep:>3}, iter {i + 1:>4}, loss: {train_loss:.6f}, "
                    f"train accuracy: {train_acc:.6f}, {sum_examples / timer.sum():.1f} examples/sec"
                )
        timer.reset()
        timer.start()
        test_acc = evaluate(net, test_iter, device)
        timer.stop()
        animator.add(ep, (None, None, test_acc))
        print(f"[{log}] test accuracy: {test_acc:.6f}, {sum_examples / timer.sum():.1f} examples/sec")
    print(f"[{log}] Completed, loss: {train_loss:.6f}, train accuracy: {train_acc:.6f}, test accuracy: {test_acc:.6f}")
    animator.save(f"./animator_{log}.jpg")


# def train_regression(net, opt, loss, data, num_epochs, log="log"):
#     animator = Animator(yscale='log', legend=['train loss'])
#     train_iter, test_iter = data.get_iter()
#     for ep in range(1, num_epochs + 1):
#         train_loss, _ = train_epoch(net, opt, loss, train_iter)
#         animator.add(ep, (train_loss))
#         print(f"[{log}] epoch {ep:>3}, train loss: {train_loss:.6f}")
#     animator.save(f"./animator_{log}.jpg")
