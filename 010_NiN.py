import torch
import torchvision
import torchsummary
from lx_module import dataset
from lx_module import optimizer
from lx_module import loss_func
from lx_module import uitls


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    # NiN 块：正常卷积+两个1*1的全连接卷积
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        torch.nn.ReLU(),
        # 两个 1*1 卷积层，相当于两层隐藏层的 MLP
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=1),
        torch.nn.ReLU(),
    )


def NiN(in_channels, num_outputs):
    # 网络中的网络 NiN 对许多后续卷积神经网络的设计产生了很大影响。
    # LeNet、AlexNet和VGG都有一个共同的设计模式：
    # 通过一系列的卷积层与汇聚层来提取空间结构特征；然后通过全连接层对特征的表征进行处理。
    # AlexNet和VGG对LeNet的改进主要在于如何扩大和加深这两个模块。

    # 全连接层需要的参数量越来越大：LeNet 48k, AlexNet 26M, VGG 102M.
    # 这么大的参数量，首先会占用大量内存，其次占用大量计算带宽，最后还很容易过拟合。
    # NiN 提供了一个非常简单的解决方案：在每个像素的通道上分别使用全连接层(1*1卷积)，
    # 然后完全取消最后的那个大的全连接层，使用一个全局平均汇聚层来生成对数几率(用于输入Softmax)。

    # NiN 块：一个卷积层 + 两个当全连接用的1*1卷积
    # NiN 架构：
    #   交替使用 NiN 块和步幅为2的最大池化层(减尺寸加通道)
    #   使用全局平均池化层来代替全连接层（参数更少，不容易过拟合）
    net = torch.nn.Sequential(
        nin_block(in_channels, 96, kernel_size=11, strides=4, padding=0),
        torch.nn.MaxPool2d(3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        torch.nn.MaxPool2d(3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        torch.nn.MaxPool2d(3, stride=2),
        nin_block(384, num_outputs, kernel_size=3, strides=1, padding=1),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),  # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    )

    # [xavier 初始化] 参数初始化函数: 当 m 是 torch.nn.Linear 权重时执行 xavier 初始化
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)
    net = net.to(device=uitls.try_gpu())
    # 打印网络结构, 每层的输出大小及参数量
    torchsummary.summary(net, input_size=(in_channels, 224, 224))
    return net


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.1  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 128  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 10  # 加载数据集使用的工作线程数
    resize_pipeline = [torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()]
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers, resize_pipeline)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = NiN(1, 10)  # 输出 (10 个类)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, data, num_epochs, "NiN")


if __name__ == "__main__":
    main()
