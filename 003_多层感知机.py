import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    loss_sum = 0  # 训练损失
    num_samples = 0  # 样本总数
    num_accuracy = 0  # 预测正确的样本数
    if isinstance(net, torch.nn.Module):
        net.train()  # 如果是 nn 模型, 将模型设置为训练模式
    for X, y in train_iter:
        y_hat = net(X)  # 计算梯度并更新参数
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):  # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            loss_sum += float(l) * len(y)
            num_samples += y.size().numel()
        else:  # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            loss_sum += float(l.sum())
            num_samples += y.numel()
        num_accuracy += uitls.accuracy(y_hat, y)
    # 返回训练损失和训练精度
    return loss_sum / num_samples, num_accuracy / num_samples


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.1  # (超参数)训练的学习率
    batch_size = 64  # (超参数)训练的批大小 (一次读取的数据数量)
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    num_workers = 8  # 加载数据集使用的工作线程数
    data = dataset.Dataset_FashionMNIST()
    train_iter, test_iter = data.get_iter(batch_size, num_workers)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    image_width = 28
    image_height = 28
    num_inputs = image_width * image_height  # 输入特征向量长度, 由数据集决定
    num_classes = 10  # 输出类别数量, 由数据集决定
    num_hiddens = [512, 256]  # (超参数)隐藏层的数量和大小 (如果隐藏层为空, 此多层感知机就退化为 Softmax 回归)

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = network.net_multilayer_perceptrons_custom(num_inputs, num_classes, num_hiddens)
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy_custom()
    for ep in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(net, train_iter, loss, opt)
        test_acc = uitls.evaluate_accuracy(net, test_iter)
        print(f"[custom] epoch {ep:>3}, loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
    train_acc = uitls.evaluate_accuracy(net, train_iter)
    test_acc = uitls.evaluate_accuracy(net, test_iter)
    print(f"[custom] Training completed, train accuracy: {train_acc:.4f}, test accuracy: {test_acc:.4f}\n\n")
    data.gen_preview_image(save_path="./custom.jpg", net=net)

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = network.net_multilayer_perceptrons(num_inputs, num_classes, num_hiddens)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()
    for ep in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(net, train_iter, loss, opt)
        test_acc = uitls.evaluate_accuracy(net, test_iter)
        print(f"[torch] epoch {ep:>3}, loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
    train_acc = uitls.evaluate_accuracy(net, train_iter)
    test_acc = uitls.evaluate_accuracy(net, test_iter)
    print(f"[torch] Training completed, train accuracy: {train_acc:.4f}, test accuracy: {test_acc:.4f}\n\n")
    data.gen_preview_image(save_path="./torch.jpg", net=net)


if __name__ == "__main__":
    main()
