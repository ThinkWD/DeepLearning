import torch
import torchvision
import torchsummary
from lx_module import dataset
from lx_module import optimizer
from lx_module import loss_func
from lx_module import uitls


class BasicBlock(torch.nn.Module):
    '''BasicBlock block for ResNet.'''

    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        # 附加层 conv1 -> bn1: 可以指定步幅来降低输出尺寸
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        # 附加层 conv2 -> bn2: 一般不改变尺寸，因此不使用 stride 参数
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        # 降采样：是否使用 1*1 卷积层来同步输出的形状。如果附加层改变了图像尺寸或通道数，就需要同步。
        self.downsample = None
        if strides != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, X):
        # 附加层(正常串联通路: X -> conv1 -> bn1 -> relu -> conv2 -> bn2)
        Y = torch.nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 同步输出形状
        if self.downsample is not None:
            X = self.downsample(X)
        # 残差映射(输入跳过附加层直接输出)
        Y += X  # 这里的相加是相同形状矩阵每个值对应相加，不是通道数翻倍
        return torch.nn.functional.relu(Y)


class Bottleneck(torch.nn.Module):
    '''Bottleneck block for ResNet.'''

    def __init__(self, in_channels, out_channels, strides=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        # 附加层1
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        # 附加层2
        self.conv2 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=strides)
        self.bn2 = torch.nn.BatchNorm2d(mid_channels)
        # 附加层3
        self.conv3 = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        # 降采样：是否使用 1*1 卷积层来同步输出的形状。如果附加层改变了图像尺寸或通道数，就需要同步。
        self.downsample = None
        if strides != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, X):
        # 附加层(正常串联通路: X -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3)
        Y = torch.nn.functional.relu(self.bn1(self.conv1(X)))
        Y = torch.nn.functional.relu(self.bn2(self.conv2(X)))
        Y = self.bn3(self.conv3(Y))
        # 同步输出形状
        if self.downsample is not None:
            X = self.downsample(X)
        # 残差映射(输入跳过附加层直接输出)
        Y += X  # 这里的相加是相同形状矩阵每个值对应相加，不是通道数翻倍
        return torch.nn.functional.relu(Y)


def ResNet_Layer(in_channels, out_channels, block, num_blocks, first_strides):
    '''first_strides 决定了是否宽高减半，通道加倍'''
    layers = []
    layers.append(block(in_channels=in_channels, out_channels=out_channels, strides=first_strides))
    for _ in range(1, num_blocks):
        layers.append(block(in_channels=out_channels, out_channels=out_channels))
    return layers


def ResNet(depth, in_channels, out_channels):
    # 通用起始阶段，快速降低尺寸。宽高减半两次，通道 in_channels -> 64
    initiate = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # 宽高减半
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 宽高减半
    )
    # 获取骨干网络类型
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3)),
    }
    if depth not in arch_settings:
        raise KeyError(f'invalid depth {depth} for resnet')
    block, stage_blocks = arch_settings[depth]
    # 骨干网络
    backbone = torch.nn.Sequential(
        *ResNet_Layer(64, 64, block, stage_blocks[0], 1),  # 除了第一个以外，宽高减半，通道加倍
        *ResNet_Layer(64, 128, block, stage_blocks[1], 2),
        *ResNet_Layer(128, 256, block, stage_blocks[2], 2),
        *ResNet_Layer(256, 512, block, stage_blocks[3], 2),
    )
    # 脖子
    neck = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化为 1*1
    )
    # 分类头, 加一个全连接层避免卷积层需要直接输出类别数的通道。
    head = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(512, out_channels),  # 1 * 1 * 512
    )
    net = torch.nn.Sequential(initiate, backbone, neck, head)

    # [xavier 初始化] 参数初始化函数: 当 m 是 torch.nn.Linear 权重时执行 xavier 初始化
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)
    net = net.to(device=uitls.try_gpu())
    # 打印网络结构, 每层的输出大小及参数量
    torchsummary.summary(net, input_size=(in_channels, 96, 96))
    return net


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.05  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 256  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 10  # 加载数据集使用的工作线程数
    resize_pipeline = [torchvision.transforms.Resize(96), torchvision.transforms.ToTensor()]
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers, resize_pipeline)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = ResNet(18, 1, 10)  # 输出 (10 个类)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, data, num_epochs, "ResNet")


if __name__ == "__main__":
    main()
