import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls

###########################################################################
#
#
#   在线性回归的基础上，对优化器添加了 权重衰减参数 来应对过拟合
#
#
###########################################################################


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.03  # (超参数)训练的学习率
    num_epochs = 20  # (超参数)训练遍历数据集的次数
    batch_size = 10  # (超参数)训练的批大小 (一次读取的数据数量)
    generator = 1  # 初始化参数使用的方差 (均值默认为 0)
    weight_decay = 3  # 权重衰减参数 (实际一般使用 1e-4 ~ 1e-2)
    # 定义真实最优解情况下的权重 w 和偏差 b, 并根据它们生成数据集
    true_w = torch.ones((200, 1)) * 0.01  # 0.01 乘以全 1 的向量
    true_b = 0.05
    data = dataset.Dataset_GaussianDistribution(true_w, true_b, 20, 100, batch_size)
    data.gen_preview_image(save_path=f"./preview_train.jpg")

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = network.net_linear_regression_custom(len(true_w), generator)  # 网络结构
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate, weight_decay)  # 优化器
    loss = loss_func.loss_squared_custom()  # 损失函数
    uitls.train_regression(net, opt, loss, data, num_epochs, "custom")
    w, _ = net.parameters()
    print(f"[custom] w 的 L2 范数: {torch.norm(w).item()}\n\n")

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = network.net_linear_regression(len(true_w), generator)  # 网络结构
    opt = optimizer.opt_sgd(net.parameters(), learn_rate, weight_decay)  # 优化器
    loss = loss_func.loss_squared()  # 损失函数
    uitls.train_regression(net, opt, loss, data, num_epochs, "torch")
    w, _ = net.parameters()
    print(f"[torch] w 的 L2 范数: {torch.norm(w).item()}\n\n")


if __name__ == "__main__":
    main()
