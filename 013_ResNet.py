import torchvision
from lx_module import dataset
from lx_module import optimizer
from lx_module import loss_func
from lx_module import uitls
from lx_module.ResNet import ResNet


def main():
    ### >>> 初始化数据集和超参数 <<< ###########################################
    learn_rate = 0.05  # (超参数)训练的学习率
    num_epochs = 10  # (超参数)训练遍历数据集的次数
    batch_size = 256  # (超参数)训练的批大小 (一次读取的数据数量)
    num_workers = 10  # 加载数据集使用的工作线程数
    pipeline = [torchvision.transforms.Resize(96), torchvision.transforms.ToTensor()]
    data = dataset.Dataset_FashionMNIST(batch_size, num_workers, pipeline)

    ### >>> 确定模型结构和超参数 <<< ###########################################
    net = ResNet(18, 1, 10)  # 输出 (10 个类)
    opt = optimizer.opt_sgd(net.parameters(), learn_rate)
    loss = loss_func.loss_cross_entropy()

    ### >>> 开始训练 <<< ################################
    uitls.train_classification(net, opt, loss, data, num_epochs, "ResNet")


if __name__ == "__main__":
    main()
