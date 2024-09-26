from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.1  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 64  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 8  # 加载数据集使用的工作线程数
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    image_width = 28
    image_height = 28
    num_inputs = image_width * image_height  # 输入特征向量长度, 由数据集决定
    num_classes = 10  # 输出类别数量, 由数据集决定
    num_hiddens = [256, 256]  # (超参数)隐藏层的数量和大小 (如果隐藏层为空, 此多层感知机就退化为 Softmax 回归)
    dropout = [0.2, 0.5]  # (超参数)隐藏层丢弃的概率

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = network.net_multilayer_perceptrons_custom(num_inputs, num_classes, num_hiddens, dropout)
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy_custom()
    uitls.train_classification(net, opt, loss, data, num_epochs, "custom")

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = network.net_multilayer_perceptrons(num_inputs, num_classes, num_hiddens, dropout)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()
    uitls.train_classification(net, opt, loss, data, num_epochs, "torch")


if __name__ == "__main__":
    main()
