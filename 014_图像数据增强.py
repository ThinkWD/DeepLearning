import PIL
import torchvision
from lx_module import dataset
from lx_module import optimizer
from lx_module import loss_func
from lx_module import uitls
from lx_module.ResNet import ResNet


def preview_augs(pipeline):
    num_rows = 10
    num_cols = 10
    img = PIL.Image.open('./cat1.png')
    augs = torchvision.transforms.Compose(pipeline)
    imgs = [augs(img) for _ in range(num_rows * num_cols)]
    dataset.show_images(imgs, num_rows, num_cols, save_path='./preview_augs.jpg')


def main():
    pipeline = [
        # 缩放到指定大小
        torchvision.transforms.Resize(96),
        # 随机翻转_垂直方向 (慎用, 特定数据集可用)
        # torchvision.transforms.RandomVerticalFlip(),
        # 随机翻转_水平方向
        torchvision.transforms.RandomHorizontalFlip(),
        # 随机亮度_对比度_饱和度_色调
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        # 随机裁切
        torchvision.transforms.RandomResizedCrop((96, 96), scale=(0.2, 1), ratio=(0.5, 2)),
        # 转为 tensor
        torchvision.transforms.ToTensor(),
    ]
    ### >>> 预览 pipeline 效果 <<< ###########################################
    # preview_augs(pipeline)

    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.05  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 256  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 8  # 加载数据集使用的工作线程数
    data = dataset.Dataset_CIFAR10(batch_size, num_workers, pipeline)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = ResNet(18, 3, 10)  # 输入通道数为 3, 输出类别数为 10
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, data, num_epochs, "ResNet_数据增强")


if __name__ == "__main__":
    main()
