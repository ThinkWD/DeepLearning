import torch
import torchsummary
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


class Reshape(torch.nn.Module):
    def forward(self, X):
        return X.view(-1, 1, 28, 28)


def LeNet(num_outputs):
    # 最早的完整卷积神经网络, 与手写数字数据集 MNIST 一起提出, 做分类任务.
    # 设计思想: 卷积+激活+池化 结构, 减输出尺寸, 加输出通道.
    # 直观理解: 不停把空间信息压缩变小, 然后把抽出来的压缩的信息放到不同的通道中.
    # 虽然 ReLU 和最大池化层更有效, 但它们在20世纪90年代还没有出现.
    net = torch.nn.Sequential(
        # 预处理, 将输入裁剪到单通道 28*28, 以适应不同的数据集
        Reshape(),
        # 第一部分, 卷积+激活+池化, 减输出尺寸, 加输出通道
        torch.nn.Conv2d(1, 6, kernel_size=5, padding=2),  # [6, 28, 28] (填充 2，宽高不变)
        torch.nn.Sigmoid(),
        torch.nn.AvgPool2d(kernel_size=2, stride=2),  # [6, 14, 14] (步幅 2, 宽高减半)
        # 第二部分, 卷积+激活+池化, 减输出尺寸, 加输出通道
        torch.nn.Conv2d(6, 16, kernel_size=5),  # [16, 10, 10] (无填充，宽高减 4)
        torch.nn.Sigmoid(),
        torch.nn.AvgPool2d(kernel_size=2, stride=2),  # [16, 5, 5] (步幅 2, 宽高减半)
        # 第三部分, 展平+MLP, 输出分类结果
        torch.nn.Flatten(),  # 16 * 5 * 5 = 400 (将所有通道的数据展平成一维向量)
        torch.nn.Linear(16 * 5 * 5, 120),  # 120
        torch.nn.Sigmoid(),
        torch.nn.Linear(120, 84),  # 84
        torch.nn.Sigmoid(),
        torch.nn.Linear(84, num_outputs),  # 类别数量
    )

    # [xavier 初始化] 参数初始化函数: 当 m 是 torch.nn.Linear 权重时执行 xavier 初始化
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)
    net = net.to(device=uitls.try_gpu())

    # 打印网络结构, 每层的输出大小及参数量
    torchsummary.summary(net, input_size=(1, 28, 28))
    return net


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.9  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 64  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 10  # 加载数据集使用的工作线程数
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = LeNet(10)  # 输出 (10 个类)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, data, num_epochs, "LeNet")


if __name__ == "__main__":
    main()
