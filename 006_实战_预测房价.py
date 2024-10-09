from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls
import pandas
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


def log_rmse(net, loss, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


def train_regression_k_fold(net, opt, loss, data, K, i, num_epochs, log="log"):
    animator = uitls.Animator(yscale='log', legend=['train loss', 'test loss'])
    train_iter, test_iter = data.get_k_fold_data_iter(K, i)
    for ep in range(1, num_epochs + 1):
        test_log_loss = uitls.log_rmse(net, loss, test_iter)
        train_log_loss = uitls.log_rmse(net, loss, train_iter)
        uitls.train_epoch(net, opt, loss, train_iter)
        animator.add(ep, (train_log_loss, test_log_loss))
        # print(
        #     f"[{log}] [{K} fold {i}] epoch {ep:>3}, "
        #     f"train loss: {train_log_loss:.6f}, test loss: {test_log_loss:.6f}, "
        # )
    test_log_loss = uitls.log_rmse(net, loss, test_iter)
    train_log_loss = uitls.log_rmse(net, loss, train_iter)
    # animator.add(num_epochs + 1, (train_log_loss, test_log_loss))
    print(
        f"[{log}] [{K} fold {i}] Training completed, "
        f"train loss: {train_log_loss:.6f}, test loss: {test_log_loss:.6f}"
    )
    animator.save(f"./animator_{log}_{K}_fold_{i}.jpg")
    return train_log_loss, test_log_loss


def main(K_fold, learn_rate, batch_size, num_epochs, weight_decay, num_hiddens, dropout):
    num_inputs = 331  # 输入特征向量长度, 由数据集决定
    num_outputs = 1  # 输出向量长度, 回归任务输出单个数值
    num_workers = 0  # 加载数据集使用的工作线程数
    data = dataset.Dataset_HousePricesAdvanced(batch_size, num_workers)
    ### >>> 使用 torch API 训练模型 <<< ################################
    loss = loss_func.loss_squared()
    train_l_sum, test_l_sum = 0, 0
    for i in range(K_fold):
        net = network.net_multilayer_perceptrons(num_inputs, num_outputs, num_hiddens, dropout)
        opt = optimizer.opt_adam(net.parameters(), learn_rate, weight_decay)
        train_l, test_l = train_regression_k_fold(net, opt, loss, data, K_fold, i, num_epochs, "torch")
        train_l_sum += train_l
        test_l_sum += test_l
    train_l_avg = float(train_l_sum / K_fold)
    test_l_avg = float(test_l_sum / K_fold)
    print(
        f'[torch]\t{K_fold}-折验证, 学习率: {learn_rate}, 批大小: {batch_size}, 总轮数: {num_epochs}, 权重衰减: {weight_decay}'
        f'\n\t隐藏层: {num_hiddens}, dropout: {dropout}\n\t结果: 平均训练 log: {train_l_avg:f}, 平均测试 log: {test_l_avg:f}\n'
    )


def train_and_pred(learn_rate, batch_size, num_epochs, weight_decay, num_hiddens, dropout, log="log"):
    num_inputs = 331  # 输入特征向量长度, 由数据集决定
    num_outputs = 1  # 输出向量长度, 回归任务输出单个数值
    num_workers = 0  # 加载数据集使用的工作线程数
    data = dataset.Dataset_HousePricesAdvanced(batch_size, num_workers)
    ### >>> 使用 torch API 训练模型 <<< ################################
    net = network.net_multilayer_perceptrons(num_inputs, num_outputs, num_hiddens, dropout)
    opt = optimizer.opt_adam(net.parameters(), learn_rate, weight_decay)
    loss = loss_func.loss_squared()
    train_iter = data.get_train_data_iter()
    for ep in range(1, num_epochs + 1):
        train_log_loss = uitls.log_rmse(net, loss, train_iter)
        uitls.train_epoch(net, opt, loss, train_iter)
        print(f"[{log}] epoch {ep:>3}, train loss: {train_log_loss:.6f}")
    train_log_loss = uitls.log_rmse(net, loss, train_iter)
    print(f"[{log}] Training completed, train loss: {train_log_loss:.6f}")
    ### >>> 导出推理结果 <<< ################################
    # 将网络应用于测试集
    test_data, test_X = data.get_test_data()
    test_X = test_X.to(uitls.try_gpu())
    preds = net(test_X).detach().cpu().numpy()
    # 将其重新格式化以导出到 Kaggle
    test_data['SalePrice'] = pandas.Series(preds.reshape(1, -1)[0])
    submission = pandas.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


# 最佳成绩: 0.13006
if __name__ == "__main__":
    K_fold = 5  # (超参数) K 折交叉验证
    learn_rate = 0.0008  # (超参数)训练的学习率
    batch_size = 256  # (超参数)训练的批大小 (一次读取的数据数量)
    num_epochs = 50  # (超参数)训练遍历数据集的次数
    weight_decay = 0.01  # 权重衰减参数 (实际一般使用 1e-4 ~ 1e-2)
    num_hiddens = [512, 1024, 512, 256, 128, 64, 32]  # (超参数)隐藏层的数量和大小
    dropout = [0.25, 0.5, 0.25, 0.1, 0, 0, 0]  # (超参数)隐藏层丢弃的概率

    # main(K_fold, learn_rate, batch_size, num_epochs, weight_decay, num_hiddens, dropout)

    train_and_pred(learn_rate, batch_size, num_epochs, weight_decay, num_hiddens, dropout)
