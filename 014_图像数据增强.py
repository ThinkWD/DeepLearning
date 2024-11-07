import PIL
import torch
import torchvision
from lx_module import dataset
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
    ### >>> 预览 augs 效果 <<< ###########################################
    # preview_augs(train_augs)

    train_augs = torchvision.transforms.Compose(
        [
            # 这个数据集尺寸太小了，经过多次宽高减半后变得非常小，不利于训练，因此需要使用 Resize 放大一下
            torchvision.transforms.Resize(96),  # 缩放到 96 * 96
            torchvision.transforms.RandomResizedCrop(96),  # 随机裁切
            # torchvision.transforms.RandomVerticalFlip(), # 垂直方向翻转
            torchvision.transforms.RandomHorizontalFlip(),  # 水平方向翻转
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
        ]
    )
    test_augs = torchvision.transforms.Compose([torchvision.transforms.Resize(96), torchvision.transforms.ToTensor()])

    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.001  # (超参数)训练的学习率
    num_epochs = 50  # (超参数)训练遍历数据集的次数
    batch_size = 256  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 8  # 加载数据集使用的工作线程数
    data = dataset.Dataset_CIFAR10(batch_size, num_workers, train_augs, test_augs)
    train_iter, test_iter = data.get_iter()

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = ResNet(18, 3, 10)  # 输入通道数为 3, 输出类别数为 10
    opt = torch.optim.Adam(net.parameters(), learn_rate)
    loss = torch.nn.CrossEntropyLoss()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, train_iter, test_iter, num_epochs, "ResNet_数据增强")


if __name__ == "__main__":
    main()
