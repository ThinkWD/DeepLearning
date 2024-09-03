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
        net.eval()  # 将模型设置为评估模式
    num_samples = 0  # 样本总数
    num_accuracy = 0  # 预测正确的样本数
    with torch.no_grad():
        for X, y in data_iter:
            num_samples += y.numel()  # 这个批次的样本总数
            num_accuracy += accuracy(net(X), y)  # 这个批次预测正确的数量
    return num_accuracy / num_samples  # 分类正确的样本数 / 总样本数 = 精度
