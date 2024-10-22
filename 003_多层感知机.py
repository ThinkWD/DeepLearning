import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


###########################################################################
#
#
#   第一个非线性模型，由隐藏层的激活函数引入非线性。
#   添加了隐藏层的 Softmax 分类模型（去掉隐藏层后退化为 Softmax 分类）
#
#
###########################################################################
def net_multilayer_perceptrons(num_inputs, num_outputs, num_hiddens):
    """网络结构: 多层感知机
    Args:
        num_inputs (int): 输入特征向量的长度, 决定权重参数数量
        num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
        num_hiddens (list): 超参数. 隐藏层的数量和每层的大小
    """
    # 前处理: 将原始图像(三维)展平为向量(一维)
    layers = [torch.nn.Flatten()]
    # 创建隐藏层
    last_num_inputs = num_inputs
    for i, num_hidden in enumerate(num_hiddens):
        layers.append(torch.nn.Linear(last_num_inputs, num_hidden))  # 隐藏层: Linear 全连接层
        layers.append(torch.nn.ReLU())  # 隐藏层的激活函数
        last_num_inputs = num_hidden
    # 创建输出层. (后处理 softmax 没有被显式定义是因为 CrossEntropyLoss 中已经包含了 softmax, 不需要重复定义)
    layers.append(torch.nn.Linear(last_num_inputs, num_outputs))
    # 创建 torch.nn 模型结构
    net = torch.nn.Sequential(*layers)
    # 参数初始化函数(lambda 表达式): 当 m 是 torch.nn.Linear 权重时执行高斯分布初始化
    init_weights = lambda m: torch.nn.init.normal_(m.weight, std=0.01) if isinstance(m, torch.nn.Linear) else None
    net.apply(init_weights)
    return net.to(device=uitls.try_gpu())


class net_multilayer_perceptrons_custom(network.BaseModel):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        """网络结构: 多层感知机 的自定义实现
        Args:
            num_inputs (int): 输入特征向量的长度, 决定权重参数数量
            num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
            num_hiddens (list): 超参数. 隐藏层的数量和每层的大小
        """
        self.params = []
        # 创建隐藏层
        last_num_inputs = num_inputs
        for num_hidden in num_hiddens:
            size = (last_num_inputs, num_hidden)
            self.params.append(torch.normal(0, 0.01, size=size, requires_grad=True, device=uitls.try_gpu()))
            self.params.append(torch.zeros(num_hidden, requires_grad=True, device=uitls.try_gpu()))
            last_num_inputs = num_hidden
        # 创建输出层
        size = (last_num_inputs, num_outputs)
        self.params.append(torch.normal(0, 0.01, size=size, requires_grad=True, device=uitls.try_gpu()))
        self.params.append(torch.zeros(num_outputs, requires_grad=True, device=uitls.try_gpu()))

    def parameters(self):
        return self.params

    def __call__(self, X):
        # 前处理: 将原始图像(三维)展平为向量(一维)
        X = X.reshape((-1, self.params[0].shape[0]))
        # 隐藏层: 全连接层, 逐层应用权重、偏置、激活函数
        for i in range(0, len(self.params) - 2, 2):
            X = network.relu(torch.matmul(X, self.params[i]) + self.params[i + 1])
        # 输出层: 全连接层, 应用权重、偏置
        X = torch.matmul(X, self.params[-2]) + self.params[-1]
        return X  # 后处理: softmax 函数将预测值转为属于每个类的概率 (定义在损失函数中)


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
    num_hiddens = [512, 256]  # (超参数)隐藏层的数量和大小 (如果隐藏层为空, 此多层感知机就退化为 Softmax 回归)

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = net_multilayer_perceptrons_custom(num_inputs, num_classes, num_hiddens)
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy_custom()
    uitls.train_classification(net, opt, loss, data, num_epochs, "custom")

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = net_multilayer_perceptrons(num_inputs, num_classes, num_hiddens)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()
    uitls.train_classification(net, opt, loss, data, num_epochs, "torch")


if __name__ == "__main__":
    main()
