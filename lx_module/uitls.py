import os
import time
import torch
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def _time_():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):  # 复位
        self.total_time = 0.0
        self.tik = time.time()

    def flag(self):  # 获取当次计时时间
        new_tik = time.time()
        elapsed = new_tik - self.tik
        self.total_time += elapsed
        self.tik = new_tik
        return elapsed

    def sum(self):  # 获取计时以来的总时间
        self.flag()
        return self.total_time


def seconds_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


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
        figsize=(10.5, 7.5),
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
    if len(y.shape) > 1:  # 检查是否启用了 mixup 或 cutmix
        y = y.argmax(axis=1)  # 对混合标签取最大值的下标
    cmp = y_hat.type(y.dtype) == y  # 与真实值比较
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的数量


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


def train_batch(net, opt, loss, X, y, device=try_gpu(), idx=0, grad_accum_steps=1, mixup_fn=None, scaler=None):
    if mixup_fn is not None:
        X, y = mixup_fn(X, y)
    X = [x.to(device) for x in X] if isinstance(X, list) else X.to(device)
    y = y.to(device)
    net.train()  # 将模型设置为训练模式: 更新参数, 应用丢弃法
    # 使用 autocast 上下文和 AMP 半精度训练支持
    use_amp = scaler is not None
    with autocast(enabled=use_amp):
        y_hat = net(X)  # 前向传播, 获取预测结果
        l = loss(y_hat, y) / grad_accum_steps  # 计算小批次的平均损失
    # 使用 GradScaler 进行反向传播和优化
    if isinstance(opt, torch.optim.Optimizer):  # 计算梯度
        scaler.scale(l.mean()).backward() if use_amp else l.mean().backward()  # 反向传播, 计算梯度
        if (idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.step(opt)  # 使用 AMP 时更新缩放后的参数
                scaler.update()  # 更新缩放比例
            else:
                opt.step()  # 常规模式下直接根据梯度更新参数
            opt.zero_grad()  # 清空上次的梯度
        num_loss = float(l) * len(y)  # 更新总损失
    else:
        l.sum().backward()  # 反向传播, 计算梯度
        if (idx + 1) % grad_accum_steps == 0:
            opt(X.shape[0])  # 根据梯度更新参数
        num_loss = float(l.sum())  # 更新总损失
    num_accuracy = accuracy(y_hat, y)  # 这个批次预测正确的数量
    return num_loss, num_accuracy


###########################################################################
#
#
#   分类任务
#
#
###########################################################################
def train_classification(
    net,
    opt,
    loss_fn,
    train_iter,
    test_iter,
    num_epochs,
    log="log",
    lr_scheduler=None,
    grad_accum_steps=1,
    mixup_fn=None,
    use_amp=False,
):
    device = try_gpu()
    num_batches = len(train_iter)
    best_checkpoint, last_checkpoint = 0.5, None
    scaler = GradScaler() if use_amp else None  # 初始化 GradScaler，仅在启用 AMP 时使用
    animator = Animator(ylim=[0, 1], legend=['train acc', 'test acc', 'train loss'])
    animator_lr = Animator(legend=['lr'])
    # 添加初始学习率数值
    if lr_scheduler is not None:
        animator_lr.add(0, opt.param_groups[0]['lr'])  # lr_scheduler.get_last_lr()[0])
    timer = Timer()
    for ep in range(1, num_epochs + 1):
        # 训练集
        sum_loss, sum_train_acc, sum_examples, sum_predictions = 0, 0, 0, 0
        for i, (X, y) in enumerate(train_iter):
            l, acc = train_batch(net, opt, loss_fn, X, y, device, i, grad_accum_steps, mixup_fn, scaler)
            sum_loss += l
            sum_train_acc += acc
            sum_examples += y.shape[0]
            sum_predictions += y.numel()
            if (i + 1) % (num_batches // 5 + 1) == 0 or i == num_batches - 1:
                l = sum_loss / sum_examples
                acc = sum_train_acc / sum_predictions
                animator.add((ep - 1) + (i + 1) / num_batches, (acc, None, l))
                lr = opt.param_groups[0]['lr']  # lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else
                print(
                    f"{_time_()} - [{log}] epoch {ep:>3}, iter {i + 1:>4}, lr: {lr:.2e}, loss: {l:.6f}, train accuracy: {acc:.6f}"
                )
        # 测试集
        test_acc = evaluate(net, test_iter, device)
        animator.add(ep, (None, test_acc, None))
        eta_time = seconds_to_hms(timer.sum() / ep * (num_epochs - ep))
        print(f"{_time_()} - [{log}] test accuracy: {test_acc:.6f}, eta: {eta_time}")
        # 更新学习率
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(test_acc)
            else:
                lr_scheduler.step()
            animator_lr.add(ep, opt.param_groups[0]['lr'])  # lr_scheduler.get_last_lr()[0])
        # 保存最佳精度模型
        if test_acc > best_checkpoint:
            best_checkpoint = test_acc
            print(f'\n{_time_()} - [{log}] Saving checkpoint in epoch {ep} with accuracy {test_acc:.6f}\n')
            if last_checkpoint is not None:
                os.remove(last_checkpoint)
            last_checkpoint = f'./{log}_{ep}_{test_acc:.3f}.pth'
            torch.save(net.state_dict(), last_checkpoint)
    print(f"{_time_()} - [{log}] Completed, best test accuracy: {best_checkpoint}, Time: {seconds_to_hms(timer.sum())}")
    animator_lr.save(f"./animator_lr_{log}_{best_checkpoint:.3f}.jpg")
    animator.save(f"./animator_{log}_{best_checkpoint:.3f}.jpg")
    return best_checkpoint


###########################################################################
#
#
#   回归任务
#
#
###########################################################################
def evaluate_loss(net, loss, data_iter, using_log_loss=False, device=try_gpu()):
    """计算在指定数据集上模型的损失 (回归任务)"""
    net.eval()  # 将模型设置为评估模式: 不计算梯度, 跳过丢弃法, 性能更好
    num_loss = 0  # 训练损失
    num_samples = 0  # 样本总数
    with torch.no_grad():
        for X, y in data_iter:
            X = [x.to(device) for x in X] if isinstance(X, list) else X.to(device)
            y = y.to(device)
            y_hat = torch.clamp(net(X), 1, float('inf')) if using_log_loss else net(X)
            l = loss(torch.log(y_hat), torch.log(y)) if using_log_loss else loss(y_hat, y)
            if isinstance(loss, torch.nn.Module):
                num_loss += l * len(y)
                num_samples += y.size().numel()
            else:
                num_loss += l.sum()
                num_samples += y.numel()
    res_loss = num_loss / num_samples
    return float(torch.sqrt(res_loss)) if using_log_loss else float(res_loss)


def train_regression(
    net, opt, loss, num_epochs, train_iter, test_iter, log="log", using_log_loss=False, lr_scheduler=None
):
    device = try_gpu()
    num_batches = len(train_iter)
    best_checkpoint, last_checkpoint = 0.5, None
    animator = Animator(ylim=[0, 1], legend=['train loss', 'test loss'])
    for ep in range(1, num_epochs + 1):
        sum_train_loss = 0
        sum_examples = 0
        for i, (X, y) in enumerate(train_iter):
            l, _ = train_batch(net, opt, loss, X, y, device)
            if using_log_loss:
                continue
            sum_train_loss += l
            sum_examples += y.numel()
            if (i + 1) % (num_batches // 5 + 1) == 0 or i == num_batches - 1:
                train_loss = sum_train_loss / sum_examples
                animator.add((ep - 1) + (i + 1) / num_batches, (train_loss, None))
                lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else opt.param_groups[0]['lr']
                print(
                    f"{_time_()} - [{log}] epoch {ep:>3}, iter {i + 1:>4}, lr: {lr:.2e}, train loss: {train_loss:.6f}"
                )
        if using_log_loss:
            train_loss = evaluate_loss(net, loss, train_iter, using_log_loss, device)
            test_loss = evaluate_loss(net, loss, test_iter, using_log_loss, device) if test_iter else -1
            animator.add(ep, (train_loss, test_loss))
            lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else opt.param_groups[0]['lr']
            print(
                f"{_time_()} - [{log}] epoch {ep:>3}, lr: {lr:.2e}, train loss: {train_loss:.6f}, test loss: {test_loss:.6f}"
            )
        else:
            test_loss = evaluate_loss(net, loss, test_iter, using_log_loss, device) if test_iter else -1
            animator.add(ep, (None, test_loss))
            print(f"{_time_()} - [{log}] test loss: {test_loss:.6f}")
        # 更新学习率
        if lr_scheduler is not None:
            lr_scheduler.step()
        # 保存最佳精度模型
        if test_loss < best_checkpoint:
            best_checkpoint = test_loss
            print(f'\n{_time_()} - [{log}] Saving checkpoint in epoch {ep} with loss {test_loss:.6f}\n')
            if last_checkpoint is not None:
                os.remove(last_checkpoint)
            last_checkpoint = f'./{log}_{ep}_{test_loss:.3f}.pth'
            torch.save(net.state_dict(), last_checkpoint)
    print(f"{_time_()} - [{log}] Completed, best test loss: {test_loss:.6f}\n")
    animator.save(f"./animator_{log}.jpg")
    return train_loss, test_loss
