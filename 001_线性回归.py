import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.03  # (超参数)训练的学习率
    num_epochs = 5  # (超参数)训练遍历数据集的次数
    batch_size = 10  # (超参数)训练的批大小 (一次读取的数据数量)
    # 定义真实最优解情况下的权重 w 和偏差 b, 并根据它们生成数据集
    true_w = torch.tensor([2, -3.4, 1.5])
    true_b = 5.2
    train_data = dataset.Dataset_Gaussian_distribution(true_w, true_b, 1000, batch_size)
    train_data.gen_preview_image(save_path=f"./preview_train.jpg")
    test_data = dataset.Dataset_Gaussian_distribution(true_w, true_b, 100, batch_size)
    test_data.gen_preview_image(save_path=f"./preview_test.jpg")

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = network.net_linear_regression_custom(len(true_w))  # 网络结构
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate)  # 优化器
    loss = loss_func.loss_squared_custom()  # 损失函数
    uitls.train_regression(net, opt, loss, train_data, test_data, num_epochs, "custom")
    w, b = net.parameters()
    print(f"[custom] w: {w}, 误差: {true_w - w.reshape(true_w.shape)}")
    print(f"[custom] b: {b}, 误差: {true_b - b}\n\n")

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = network.net_linear_regression(len(true_w))  # 网络结构
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)  # 优化器
    loss = loss_func.loss_squared()  # 损失函数
    uitls.train_regression(net, opt, loss, train_data, test_data, num_epochs, "torch")
    w, b = net.parameters()
    print(f"[torch] w: {w}, 估计误差: {true_w - w.reshape(true_w.shape)}")
    print(f"[torch] b: {b}, 估计误差: {true_b - b}\n\n")


if __name__ == "__main__":
    main()
