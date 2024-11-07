import torch
import torchvision
import torchsummary
from lx_module import dataset
from lx_module import optimizer
from lx_module import network
from lx_module import loss_func
from lx_module import uitls


def AlexNet(in_channels, num_outputs):
    # AlexNet 是更大更深的 LeNet，10x 参数个数，260x 计算复杂度
    # AlexNet 引入了丢弃法，ReLU，最大池化层，和数据增强
    # AlexNet 赢下了 2012 ImageNet 竞争，标志着新的一轮神经网络热潮的开始。
    net = torch.nn.Sequential(
        # 第一部分, 卷积+激活+池化, 减输出尺寸, 加输出通道
        torch.nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        # 第二部分, 卷积+激活+池化, 减输出尺寸, 加输出通道
        torch.nn.Conv2d(96, 256, kernel_size=5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        # 第三部分
        torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(384, 384, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        # 第四部分, 展平+MLP, 输出分类结果
        torch.nn.Flatten(),  # 16 * 5 * 5 = 400 (将所有通道的数据展平成一维向量)
        torch.nn.Linear(6400, 4096),  # 120
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096, 4096),  # 84
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096, num_outputs),  # 类别数量
    )

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
    learn_rate = 0.03  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 128  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 10  # 加载数据集使用的工作线程数
    pipeline = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers, pipeline, pipeline)
    train_iter, test_iter = data.get_iter()

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = AlexNet(1, 10)  # 输出 (10 个类)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, train_iter, test_iter, num_epochs, "AlexNet")


if __name__ == "__main__":
    main()
