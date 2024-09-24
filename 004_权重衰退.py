import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.03  # (超参数)训练的学习率
    num_epochs = 5  # (超参数)训练遍历数据集的次数
    batch_size = 10  # (超参数)训练的批大小 (一次读取的数据数量)
    num_inputs = 200  # 权重的数量, 影响数据和模型结构
    # 定义真实最优解情况下的权重 w 和偏差 b, 并根据它们生成数据集
    true_w = torch.ones((num_inputs, 1)) * 0.01  # 0.01 乘以全 1 的向量
    true_b = 0.05
    train_data = dataset.Dataset_Gaussian_distribution(true_w, true_b, 20)
    train_data.gen_preview_image(save_path=f"./preview_train_data.jpg")
    test_data = dataset.Dataset_Gaussian_distribution(true_w, true_b, 100)
    test_data.gen_preview_image(save_path=f"./preview_test_data.jpg")

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = network.net_linear_regression_custom(len(true_w))  # 网络结构
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate)  # 优化器
    loss = loss_func.loss_squared_custom()  # 损失函数


if __name__ == "__main__":
    main()
