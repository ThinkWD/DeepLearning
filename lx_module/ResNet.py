import torch
import torchsummary
from .uitls import try_gpu


class BasicBlock(torch.nn.Module):
    '''BasicBlock block for ResNet.'''

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, strides: int = 1) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # 附加层1
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        # 附加层2
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        # 降采样：是否使用 1*1 卷积层来同步输出的形状。如果附加层改变了图像尺寸或通道数，就需要同步。
        self.downsample = None
        if strides != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 附加层(正常串联通路: X -> conv1 -> bn1 -> relu -> conv2 -> bn2)
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 同步输出形状
        if self.downsample is not None:
            X = self.downsample(X)
        # 残差映射(输入跳过附加层直接输出)
        Y += X  # 这里的相加是相同形状矩阵每个值对应相加，不是通道数翻倍
        return self.relu(Y)


class Bottleneck(torch.nn.Module):
    '''Bottleneck block for ResNet.'''

    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, strides: int = 1) -> None:
        super().__init__()
        mid_channels = out_channels // self.expansion
        self.relu = torch.nn.ReLU(inplace=True)
        # 附加层1
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        # 附加层2
        self.conv2 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, stride=strides, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(mid_channels)
        # 附加层3
        self.conv3 = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        # 降采样：是否使用 1*1 卷积层来同步输出的形状。如果附加层改变了图像尺寸或通道数，就需要同步。
        self.downsample = None
        if strides != 1 or in_channels != out_channels:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 附加层(正常串联通路: X -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3)
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        # 同步输出形状
        if self.downsample is not None:
            X = self.downsample(X)
        # 残差映射(输入跳过附加层直接输出)
        Y += X  # 这里的相加是相同形状矩阵每个值对应相加，不是通道数翻倍
        return self.relu(Y)


def _make_layer(in_channels, out_channels, block, num_blocks, first_strides):
    '''first_strides 决定了是否宽高减半，通道加倍'''
    layers = []
    layers.append(block(in_channels=in_channels, out_channels=out_channels, strides=first_strides))
    for _ in range(1, num_blocks):
        layers.append(block(in_channels=out_channels, out_channels=out_channels))
    return layers


def ResNet(depth: int, in_channels: int, num_classes: int, show_summary: bool = False) -> torch.nn.Sequential:
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3)),
    }
    assert depth in arch_settings, f'invalid depth {depth} for resnet'
    # 通用起始阶段，快速降低尺寸。宽高减半两次，通道 in_channels -> 64
    initiate = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 宽高减半
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 宽高减半
    )
    # 骨干网络
    block, stage_blocks = arch_settings[depth]
    backbone = torch.nn.Sequential(
        *_make_layer(64, 64 * block.expansion, block, stage_blocks[0], 1),  # 除了第一个以外，宽高减半，通道加倍
        *_make_layer(64 * block.expansion, 128 * block.expansion, block, stage_blocks[1], 2),
        *_make_layer(128 * block.expansion, 256 * block.expansion, block, stage_blocks[2], 2),
        *_make_layer(256 * block.expansion, 512 * block.expansion, block, stage_blocks[3], 2),
    )
    # 脖子
    neck = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化为 1*1
    )
    # 分类头, 加一个全连接层避免卷积层需要直接输出类别数的通道。
    head = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(512 * block.expansion, num_classes),  # 1 * 1 * 512
    )
    net = torch.nn.Sequential(initiate, backbone, neck, head)

    # 参数初始化
    def init_weights(m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    net.apply(init_weights)
    net = net.to(device=try_gpu())
    if show_summary:
        # 打印网络结构, 每层的输出大小及参数量
        torchsummary.summary(net, input_size=(in_channels, 224, 224))
        # 简单四个模块的输出
        X = torch.randn(1, in_channels, 224, 224, device=try_gpu())
        print(f'Original input shape -> {X.shape}')
        net.eval()
        for layer in net:
            X = layer(X)
            print(f'{layer.__class__.__name__:>20} -> {X.shape}')
    return net
