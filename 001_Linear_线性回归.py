import torch
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


def net_linear_regression(num_key_factors):
    """网络结构: 线性回归模型 的 torch API 实现
    Args:
        num_key_factors (int): 影响模型结果的关键因素的数量
    """
    net = torch.nn.Sequential(torch.nn.Linear(num_key_factors, 1))
    net[0].weight.data.normal_(0, 0.01)  # w
    net[0].bias.data.fill_(0)  # b
    return net.to(device=uitls.try_gpu())


class net_linear_regression_custom(network.BaseModel):
    def __init__(self, num_key_factors):
        """网络结构: 线性回归模型 的自定义实现
        Args:
            num_key_factors (int): 影响模型结果的关键因素的数量
        """
        self.w = torch.normal(0, 0.01, size=(num_key_factors, 1), requires_grad=True, device=uitls.try_gpu())
        self.b = torch.zeros(1, requires_grad=True, device=uitls.try_gpu())

    def parameters(self):
        return self.w, self.b

    def __call__(self, X):
        return torch.matmul(X, self.w) + self.b  # 没有前后处理, 只有一层输出层


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.03  # (超参数)训练的学习率
    num_epochs = 5  # (超参数)训练遍历数据集的次数
    batch_size = 10  # (超参数)训练的批大小 (一次读取的数据数量)
    # 定义真实最优解情况下的权重 w 和偏差 b, 并根据它们生成数据集
    true_w = torch.tensor([2, -3.4, 1.5])
    true_b = 5.2
    data = dataset.Dataset_GaussianDistribution(true_w, true_b, 1000, 100, batch_size)
    train_iter, test_iter = data.get_iter()
    data.gen_preview_image(save_path=f"./preview_train.jpg")

    ### >>> 使用自定义实现训练模型 <<< ################################
    net = net_linear_regression_custom(len(true_w))  # 网络结构
    opt = optimizer.opt_sgd_custom(net.parameters(), learn_rate)  # 优化器
    loss = loss_func.loss_squared_custom()  # 损失函数
    uitls.train_regression(net, opt, loss, num_epochs, train_iter, test_iter, "custom")
    w, b = net.parameters()
    w, b = w.to(device='cpu'), b.to(device='cpu')
    print(f"[custom] w: {w}, 误差: {true_w - w.reshape(true_w.shape)}")
    print(f"[custom] b: {b}, 误差: {true_b - b}\n\n")

    ### >>> 使用 torch API 训练模型 <<< ################################
    net = net_linear_regression(len(true_w))  # 网络结构
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)  # 优化器
    loss = loss_func.loss_squared()  # 损失函数
    uitls.train_regression(net, opt, loss, num_epochs, train_iter, test_iter, "torch")
    w, b = net.parameters()
    w, b = w.to(device='cpu'), b.to(device='cpu')
    print(f"[torch] w: {w}, 估计误差: {true_w - w.reshape(true_w.shape)}")
    print(f"[torch] b: {b}, 估计误差: {true_b - b}\n\n")


if __name__ == "__main__":
    main()
