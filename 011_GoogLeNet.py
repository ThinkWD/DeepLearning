import torch
import torchvision
import torchsummary
from lx_module import dataset
from lx_module import optimizer
from lx_module import loss_func
from lx_module import uitls


class Inception(torch.nn.Module):
    # Inception 很可能得名于电影《盗梦空间》，因为电影中的一句话 "We need to go deeper"。
    # Inception 块解决了多大的卷积核最合适的问题，具体思路就是小学生才做选择，而我全都要。
    #                         ┏━━━━━━━━━━━━━━━━━━━━━┓
    #         ┌──────────────>┃  Concatenation(256) ┃<──────────────┐
    #         │               ┗━━━━━━━━━━━━━━━━━━━━━┛               │
    #         │                    ↑           ↑                    │
    #         │         ┏━━━━━━━━━━━━━━━┓ ┏━━━━━━━━━━━━━━━┓ ┏━━━━━━━━━━━━━━━┓
    #         │         ┃ 3x3 Conv(128) ┃ ┃ 5x5 Conv(32)  ┃ ┃ 1x1 Conv(32)  ┃
    # ┏━━━━━━━━━━━━━━━┓ ┗━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━┛
    # ┃ 1x1 Conv(64)  ┃         ↑                 ↑                 ↑
    # ┗━━━━━━━━━━━━━━━┛ ┏━━━━━━━━━━━━━━━┓ ┏━━━━━━━━━━━━━━━┓ ┏━━━━━━━━━━━━━━━┓
    #         ↑         ┃ 1x1 Conv(96)  ┃ ┃ 1x1 Conv(16)  ┃ ┃  3x3 MaxPool  ┃
    #         │         ┗━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━━━┛
    #         │                    ↑           ↑                    ↑
    #         │                  ┏━━━━━━━━━━━━━━━┓                  │
    #         └──────────────────┃   Input(192)  ┃──────────────────┘
    #                            ┗━━━━━━━━━━━━━━━┛
    # 它将输入复制 4 份，送入四个线路，每个线路使用不同大小的卷积核从不同空间大小中提取信息。
    # 每个线路中都有一个 1x1 卷积层可以改变通道数，并且用合适的填充使输入输出的尺寸一致。
    # 每个线路输出的通道数是超参数，输出时会在通道维度上将四条线路的结果组合在一起。
    #
    # 反直觉的是，这个复杂结构的参数量和计算复杂度反而比使用单纯的 3*3 或 5*5 卷积层要低的多！
    # Inception 块不仅增加了特征提取的多样性，而且参数变少，计算量还变低了。
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):  # c1~c4是每条路径的输出通道数
        super(Inception, self).__init__(**kwargs)
        # 线路1，单 1x1 卷积层
        self.p1_1 = torch.nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1 卷积层后接 3x3 卷积层
        self.p2_1 = torch.nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = torch.nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1 卷积层后接 5x5 卷积层
        self.p3_1 = torch.nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = torch.nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3 最大汇聚层后接 1x1 卷积层
        self.p4_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = torch.nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        # Inception 块不改变输入输出尺寸，只改变通道数。
        p1 = torch.nn.functional.relu(self.p1_1(x))
        p2 = torch.nn.functional.relu(self.p2_2(torch.nn.functional.relu(self.p2_1(x))))
        p3 = torch.nn.functional.relu(self.p3_2(torch.nn.functional.relu(self.p3_1(x))))
        p4 = torch.nn.functional.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


def GoogLeNet(in_channels, num_outputs):
    # GoogLeNet 中 google 的 l 大写是为了致敬 LeNet
    # GoogLeNet 吸收了 NiN 中串联网络的思想，并在此基础上做了改进。
    # GoogLeNet 提出了 Inception 块，解决了什么样大小的卷积核最合适的问题。
    # GoogLeNet 由于使用了 Inception 块，模型参数较少，计算复杂度较低。
    # GoogLeNet 使用了 9 个 Inception 块，是第一个达到上百层的网络。
    # 它的缺点是过于复杂，超参数太多，因此不是特别受欢迎
    # https://blog.csdn.net/shuzfan/article/details/50738394

    # stage1: 通用起始阶段，快速降低尺寸。宽高减半两次，通道 in_channels -> 64
    stage1 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # 宽高减半
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 宽高减半
    )
    # stage2: 宽高减半，通道 64 -> 192
    stage2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 通道 64 -> 192
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 宽高减半
    )
    # stage3: 宽高减半，通道 192 -> 256 -> 480
    stage3 = torch.nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),  # 输入 192 通道，输出 64+128+32+32=256 通道
        Inception(256, 128, (128, 192), (32, 96), 64),  # 输入 256 通道，输出 128+192+96+64=480 通道
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 宽高减半
    )
    # stage4: 宽高减半，通道 480 -> 512 -> 512 -> 512 -> 528 -> 832
    stage4 = torch.nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),  # 输入 480 通道，输出 512 通道
        Inception(512, 160, (112, 224), (24, 64), 64),  # 输入 512 通道，输出 512 通道
        Inception(512, 128, (128, 256), (24, 64), 64),  # 输入 512 通道，输出 512 通道
        Inception(512, 112, (144, 288), (32, 64), 64),  # 输入 512 通道，输出 528 通道
        Inception(528, 256, (160, 320), (32, 128), 128),  # 输入 528 通道，输出 832 通道
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 宽高减半
    )
    # stage5: 宽高减半，通道 832 -> 832 -> 1024
    stage5 = torch.nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),  # 输入 832 通道，输出 832 通道
        Inception(832, 384, (192, 384), (48, 128), 128),  # 输入 832 通道，输出 1024 通道
        torch.nn.AdaptiveAvgPool2d((1, 1)),  # 宽高减半 5 次，得到 7*7，最后被平均池化为 1*1
        torch.nn.Flatten(),
    )
    # 分类头，加一个全连接层避免卷积层需要直接输出类别数的通道。
    clas_head = torch.nn.Sequential(
        torch.nn.Linear(1024, num_outputs),  # 1 * 1 * 1024
    )
    # 组合完整网络
    net = torch.nn.Sequential(stage1, stage2, stage3, stage4, stage5, clas_head)

    # [xavier 初始化] 参数初始化函数: 当 m 是 torch.nn.Linear 权重时执行 xavier 初始化
    def init_weights(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device=uitls.try_gpu())
    # 打印网络结构, 每层的输出大小及参数量
    torchsummary.summary(net, input_size=(in_channels, 96, 96))
    return net


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.1  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 128  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 10  # 加载数据集使用的工作线程数
    resize_pipeline = [torchvision.transforms.Resize(96), torchvision.transforms.ToTensor()]
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers, resize_pipeline)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = GoogLeNet(1, 10)  # 输出 (10 个类)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, data, num_epochs, "GoogLeNet")


if __name__ == "__main__":
    main()
