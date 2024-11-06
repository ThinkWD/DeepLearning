import torch
import torchvision
import torchsummary
from lx_module import dataset
from lx_module import uitls


def main():
    # 使用RGB通道的均值和标准差，以标准化每个通道
    normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_augs = [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.2, 1), ratio=(0.5, 2)),
        torchvision.transforms.ToTensor(),
        normalize,
    ]
    test_augs = [
        torchvision.transforms.Resize([256, 256]),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ]

    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 5e-4  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 256  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 8  # 加载数据集使用的工作线程数
    data = dataset.Dataset_CIFAR10(batch_size, num_workers, train_augs, test_augs)

    ### >>> 确定模型结构 <<< ###########################################
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = torch.nn.Linear(net.fc.in_features, 10)
    torch.nn.init.xavier_uniform_(net.fc.weight)
    net = net.to(device=uitls.try_gpu())
    torchsummary.summary(net, input_size=(3, 224, 224))

    ### >>> 开始训练 <<< ################################
    params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
    opt = torch.optim.SGD(
        [{'params': params_1x}, {'params': net.fc.parameters(), 'lr': learn_rate * 10}],
        lr=learn_rate,
        weight_decay=0.001,
    )
    loss = torch.nn.CrossEntropyLoss()
    uitls.train_classification(net, opt, loss, data, num_epochs, "ResNet_with_lr_trick")


if __name__ == "__main__":
    main()
