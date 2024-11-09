import torch
import torchvision
from lx_module import uitls
from lx_module import dataset
from lx_module.ResNet import ResNet


# 数据方面：
# - 手动去除重复图片
# - 图片上背景较多，且树叶没有方向性，可以做更多增强
# 	- 随机旋转，更大的裁剪
# - 跨图片增强：
# 	- Mixup：随机叠加两张图片
# 	- CutMix：随机组合来自不同图片的块
# - 测试时使用稍弱的增强，然后多个结果投票
#
# 模型方面：
# - 模型多为 ResNet 的变种
# 	- DenseNet, ResNeXt, ResNeSt, ...
# 	- EfficientNet
# - 优化算法多为 Adam 或其变种
# - 学习率一般是 Cosine 或者训练不动时往下调
# - 训练多个模型，最后进行结果投票


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


def get_model(K_flod, flod):
    num_classes = 176
    if flod < K_flod:
        net = torchvision.models.resnext50_32x4d(pretrained=True)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        torch.nn.init.xavier_uniform_(net.fc.weight)
        net = net.to(device=uitls.try_gpu())
        logger = f'resnext50_32x4d-fold-{flod % K_flod}'
        return net, 64, 1e-4, logger
    elif flod < K_flod * 2:
        net = torchvision.models.densenet121(pretrained=True)
        net.classifier = torch.nn.Linear(net.classifier.in_features, num_classes)
        torch.nn.init.xavier_uniform_(net.classifier.weight)
        net = net.to(device=uitls.try_gpu())
        logger = f'densenet121-fold-{flod % K_flod}'
        return net, 32, 1e-4, logger
    elif flod < K_flod * 3:
        net = ResNet(50, 3, num_classes)
        logger = f'resnet50-fold-{flod % K_flod}'
        return net, 64, 1e-4, logger
    elif flod < K_flod * 4:
        net = ResNet(101, 3, num_classes)
        logger = f'resnet101-fold-{flod % K_flod}'
        return net, 32, 1e-4, logger
    else:
        raise 'no define'


def main():
    # trick: 图片上背景较多，且树叶没有方向性，可以做更多增强.
    # [未应用] trick: 跨图片增强 Mixup CutMix
    train_augs = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomResizedCrop(224),  # 随机裁切
            torchvision.transforms.RandomVerticalFlip(),  # 垂直方向翻转
            torchvision.transforms.RandomHorizontalFlip(),  # 水平方向翻转
            torchvision.transforms.RandomRotation(45),  # 随机旋转 (-45° ~ 45°)
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # hue=0.1
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # trick: 测试时使用稍弱的增强，然后多个结果投票
    test_augs = torchvision.transforms.Compose(
        [
            # torchvision.transforms.Resize(256),  # 先缩放到一个较大的尺寸
            # torchvision.transforms.FiveCrop(224),  # 上下左右中心裁剪, 获得 5 张图片
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ### >>> 数据集 <<< ######################################################################
    data = dataset.Dataset_classify_leaves(32, 8, train_augs, test_augs)
    loss = torch.nn.CrossEntropyLoss()
    ### >>> 训练 <<< ########################################################################
    K_flod = 5  # K 折交叉验证
    num_epochs = 30  # 训练轮数
    total_acc = 0  # 计算K折平均精度
    for flod in range(K_flod * 2):
        # Get the model
        net, batch_size, learn_rate, logger = get_model(K_flod, flod)
        print(f'\n\nstart training {logger} with batch_size {batch_size}')
        # Train the model
        train_iter, test_iter = data.get_k_fold_data_iter(K_flod, flod % K_flod, batch_size)
        opt = torch.optim.Adam(net.parameters(), learn_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)  # trick: 使用 Cosine 学习率策略
        total_acc += uitls.train_classification(net, opt, loss, train_iter, test_iter, num_epochs, logger, scheduler)
        if (flod + 1) % K_flod == 0:
            print(f'\n\n\n{logger} Average test accuracy: {total_acc / K_flod:.6f}\n\n\n')
            total_acc = 0
    # 继续提高精度方向：使用跨图片增强 Mixup CutMix，测试时使用 FiveCrop 投票，增大网络输入尺寸，增加更多模型


if __name__ == "__main__":
    main()
