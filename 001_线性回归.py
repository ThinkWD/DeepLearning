import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func


def train_epoch(net, train_iter, loss, opt):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式
    loss_sum = 0  # 训练损失
    num_samples = 0  # 样本总数
    for X, y in train_iter:
        y_hat = net(X)  # 计算梯度并更新参数
        l = loss(y_hat, y)
        if isinstance(opt, torch.optim.Optimizer):
            opt.zero_grad()
            l.mean().backward()  # 使用PyTorch内置的优化器和损失函数
            opt.step()
        else:
            l.sum().backward()  # 使用定制的优化器和损失函数
            opt(X.shape[0])
        loss_sum += float(l.sum())
        num_samples += y.numel()
    return loss_sum / num_samples


def main():
    ### >>> 生成数据集 <<< ###########################################
    # 定义真实最优解情况下的权重 w 和偏差 b, 并根据它们生成数据集
    true_w = torch.tensor([2, -3.4, 1.5])
    true_b = 5.2
    data = dataset.Dataset_Gaussian_distribution(true_w, true_b, 1000)
    # data.save_preview_image()

    ### >>> 设置超参数 <<< ###########################################
    lr = 0.03
    num_epochs = 5
    batch_size = 10

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = network.net_linear_regression_custom(len(true_w))  # 网络结构
    opt = optimizer.opt_sgd_custom(net.parameters(), lr)  # 优化器
    loss = loss_func.loss_squared_custom()  # 损失函数
    for epoch in range(num_epochs):  # 开始训练
        train_metrics = train_epoch(net, data.get_iter(batch_size), loss, opt)
        print(f"[custom] epoch {epoch + 1}, loss {float(train_metrics):f}")
    w, b = net.parameters()
    print(f"[custom] w: {w}, 误差: {true_w - w.reshape(true_w.shape)}")
    print(f"[custom] b: {b}, 误差: {true_b - b}\n\n")

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = network.net_linear_regression(len(true_w))  # 网络结构
    opt = optimizer.opt_sgd(net.parameters(), lr)  # 优化器
    loss = loss_func.loss_squared()  # 损失函数
    for epoch in range(num_epochs):  # 开始训练
        train_metrics = train_epoch(net, data.get_iter(batch_size), loss, opt)
        print(f"[torch] epoch {epoch + 1}, loss {float(train_metrics):f}")
    w, b = net.parameters()
    print(f"[torch] w: {w}, 估计误差: {true_w - w.reshape(true_w.shape)}")
    print(f"[torch] b: {b}, 估计误差: {true_b - b}\n\n")


if __name__ == "__main__":
    main()
