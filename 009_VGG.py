import torch
import torchvision
import torchsummary
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


def vgg_block(num_convs, in_channels, out_channels):
    # VGG 块：卷积+激活+池化 经典组合
    layers = []
    for _ in range(num_convs):
        layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        in_channels = out_channels
    layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


def VGG(in_channels, num_outputs):
    # VGG 是更大更深的 AlexNet。整个神经网络的发展历程就是更大更深。
    # AlexNet 也是比 LeNet 更深更大来得到更好的精度。所以我们需要更加深、更加大。
    # 更深更大的选择：更多的全连接层(太贵)、更多卷积层、将卷积层组合成块。
    # 神经网络是更深好还是更宽好？5*5卷积，3*3卷积？结论是深但窄效果更好。
    # VGG 使用可重复使用的卷积块(VGG块)来构建深度卷积神经网络。
    # VGG 块：3*3卷积(填充1，n层m通道可调)，2*2池化(步幅2)。
    # VGG 架构：多个 VGG 块后接全连接层。不同次数的重复块得到不同的架构，VGG-16, VGG-19。

    # VGG 块结构定义。设计思想就是分成几个块，每个块都是宽高减半，通道翻倍。(除了最后一块)
    VGG_19 = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))  # 标准 VGG-19 结构
    VGG_16 = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # 标准 VGG-16 结构
    VGG_11 = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))  # 自定义的 'VGG-11' 结构，用于演示
    layers = []
    last_in_channels = in_channels
    for num_convs, out_channels in VGG_11:
        layers.extend(vgg_block(num_convs, last_in_channels, out_channels))
        last_in_channels = out_channels
    # 最后的分类器，展平+MLP, 输出分类结果
    classifier = [
        torch.nn.Flatten(),
        torch.nn.Linear(out_channels * 7 * 7, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096, num_outputs),  # 类别数量
    ]
    layers.extend(classifier)
    net = torch.nn.Sequential(*layers)

    # [xavier 初始化] 参数初始化函数: 当 m 是 torch.nn.Linear 权重时执行 xavier 初始化
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device=uitls.try_gpu())
    # 打印网络结构, 每层的输出大小及参数量
    torchsummary.summary(net, input_size=(in_channels, 224, 224))
    return net


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.05  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 128  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 10  # 加载数据集使用的工作线程数
    pipeline = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers, pipeline, pipeline)
    train_iter, test_iter = data.get_iter()

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = VGG(1, 10)  # 输出 (10 个类)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, train_iter, test_iter, num_epochs, "VGG")


if __name__ == "__main__":
    main()
