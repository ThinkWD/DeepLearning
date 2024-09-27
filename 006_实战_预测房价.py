from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls
import torch

###########################################################################
#
#
#   kaggle 房价预测比赛
#   网址：https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#   数据集：./dataset/HousePricesAdvanced
#
#
###########################################################################


# def log_rmse(net, features, labels):
#     # 对 y 和 y_hat 都做个 log, 然后再计算损失
#     # 为了在取对数时进一步稳定该值，将小于1的值设置为1
#     clipped_preds = torch.clamp(net(features), 1, float('inf'))
#     rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
#     return rmse.item()


def train_regression(net, opt, loss, data, num_epochs, log="log"):
    animator = uitls.Animator(legend=['train loss'])  # yscale='log',
    train_iter, test_iter = data.get_iter()
    for ep in range(1, num_epochs + 1):
        train_loss, _ = uitls.train_epoch(net, opt, loss, train_iter)
        animator.add(ep, train_loss)
        print(f"[{log}] epoch {ep}\t, train loss: {train_loss:.6f}")
    train_loss, _ = uitls.evaluate(net, loss, train_iter)
    animator.add(num_epochs + 1, train_loss)
    print(f"[{log}] Training completed, train loss: {train_loss:.6f}")


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.03  # (超参数)训练的学习率
    num_epochs = 20  # (超参数)训练遍历数据集的次数
    batch_size = 64  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 8  # 加载数据集使用的工作线程数
    weight_decay = 0.001  # 权重衰减参数 (实际一般使用 1e-4 ~ 1e-2)
    data = dataset.Dataset_HousePricesAdvanced(batch_size, num_workers)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    num_inputs = 79  # 输入特征向量长度, 由数据集决定
    num_hiddens = []  # (超参数)隐藏层的数量和大小
    dropout = []  # (超参数)隐藏层丢弃的概率
    net = network.net_multilayer_perceptrons_custom(num_inputs, 1, num_hiddens, dropout)
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate, weight_decay)
    loss = loss_func.loss_squared_custom()
    train_regression(net, opt, loss, data, num_epochs, "custom")


if __name__ == "__main__":
    main()
