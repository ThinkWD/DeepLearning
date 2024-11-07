import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


###########################################################################
#
#
#   在线性回归的基础上，更换损失函数，添加前后处理，将回归任务转为分类任务
#
#
###########################################################################
def net_softmax_regression(num_inputs, num_outputs):
    """网络结构: softmax 回归
    Args:
        num_inputs (int): 输入特征向量的长度, 决定权重参数数量
        num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
    """
    net = torch.nn.Sequential(
        torch.nn.Flatten(),  # 前处理: 将原始图像(三维)展平为向量(一维)
        torch.nn.Linear(num_inputs, num_outputs),  # 输出层: Linear 全连接层
        # 后处理 softmax 没有被显式定义是因为 CrossEntropyLoss 中已经包含了 softmax, 不需要重复定义
    )
    net[1].weight.data.normal_(0, 0.01)  # w
    net[1].bias.data.fill_(0)  # b
    return net.to(device=uitls.try_gpu())


class net_softmax_regression_custom(network.BaseModel):
    def __init__(self, num_inputs, num_outputs):
        """网络结构: Softmax 回归 的自定义实现
        Args:
            num_inputs (int): 输入特征向量的长度, 决定权重参数数量
            num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
        """
        # 权重 w 使用高斯分布(均值0方差0.01) 初始化为随机值, 偏差 b 初始化为 0
        self.w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True, device=uitls.try_gpu())
        self.b = torch.zeros(num_outputs, requires_grad=True, device=uitls.try_gpu())

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, X):
        X = X.reshape((-1, self.w.shape[0]))  # 前处理: 将原始图像(三维)展平为向量(一维)
        X = torch.matmul(X, self.w) + self.b  # 输出层: Linear 全连接层
        return X  # 后处理: softmax 函数将预测值转为属于每个类的概率 (定义在损失函数中)


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.1  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 64  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 8  # 加载数据集使用的工作线程数
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers)
    train_iter, test_iter = data.get_iter()

    ### >>> 确定模型结构和超参数 <<< ###########################################
    image_width = 28
    image_height = 28
    num_inputs = image_width * image_height  # 输入特征向量长度, 由数据集决定
    num_classes = 10  # 输出类别数量, 由数据集决定

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = net_softmax_regression_custom(num_inputs, num_classes)
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy_custom()
    uitls.train_classification(net, opt, loss, train_iter, test_iter, num_epochs, "custom")

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = net_softmax_regression(num_inputs, num_classes)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()
    uitls.train_classification(net, opt, loss, train_iter, test_iter, num_epochs, "torch")


if __name__ == "__main__":
    main()
