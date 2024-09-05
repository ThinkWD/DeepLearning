import time
import torch
import random
import numpy as np
import torchvision  # 对于计算机视觉实现的一个库
import matplotlib.pyplot as plt

###########################################################
#
#  数据集 自定义实现
#
###########################################################


defult_pipeline = [torchvision.transforms.ToTensor()]
resize_pipeline = [
    torchvision.transforms.Resize(64),
    torchvision.transforms.ToTensor(),
]


class Dataset_Gaussian_distribution(object):
    def __init__(self, true_w, true_b, num_examples):
        """根据预定义的真实权重和偏差生成专用于线性回归模型,
            带有随机噪声的高斯分布数据集 (y = X * w + b + 噪声)

        Args:
            weight (向量): 线性回归模型的权重.
            deviation (标量): 线性回归模型的偏差.
            num_examples (int): 要生成的样本总数.

        Returns:
            features (矩阵, 形状为[num_examples, len(weight)]): 生成的数据.
            labels (向量, 长度为 num_examples): 生成的数据对应的标签.
        """
        # 生成形状为 (num_examples, len(w)) 的矩阵, 矩阵用均值为 0 方差为 1 的随机数来填充
        self.features = torch.normal(0, 1, (num_examples, len(true_w)))
        # 生成 labels 为矩阵 features 乘 w, 最后整体加上偏差 b
        self.labels = torch.matmul(self.features, true_w) + true_b
        # 为了让这个问题变得难一点, 给 labels 添加一些噪声
        self.labels += torch.normal(0, 0.01, self.labels.shape)
        # 将 labels 转为列向量
        self.labels = self.labels.reshape((-1, 1))

    def get_dataset(self):
        return self.features, self.labels

    def get_iter(self, batch_size, num_workers=1):
        data_arrays = (self.features, self.labels)
        dataset = torch.utils.data.TensorDataset(*data_arrays)
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    def gen_preview_image(self, save_path=None):
        plt.scatter(self.features[:, 1].detach().numpy(), self.labels.detach().numpy(), 1)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# 一个简单的服装分类数据集
class Dataset_FashionMNIST(object):
    def __init__(self, pipeline=defult_pipeline, save_path="./dataset"):
        self.text_labels = [
            't-shirt',
            'trouser',
            'pullover',
            'dress',
            'coat',
            'sandal',
            'shirt',
            'sneaker',
            'bag',
            'ankle boot',
        ]
        # 初始化 pipeline.
        transforms = torchvision.transforms.Compose(pipeline)
        # 通过内置函数下载数据集到 save_path 目录下
        self.train = torchvision.datasets.FashionMNIST(root=save_path, train=True, transform=transforms, download=True)
        self.test = torchvision.datasets.FashionMNIST(root=save_path, train=False, transform=transforms, download=True)

    def get_iter(self, batch_size, num_workers):
        train = torch.utils.data.DataLoader(self.train, batch_size, shuffle=True, num_workers=num_workers)
        test = torch.utils.data.DataLoader(self.train, batch_size, shuffle=False, num_workers=num_workers)
        return train, test

    def gen_preview_image(self, save_path=None, num_rows=5, num_cols=5, scale=1.5, net=None):
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i in range(num_rows):
            for j in range(num_cols):
                img = self.train[i * num_cols + j][0]
                if torch.is_tensor(img):
                    img = img.numpy()
                axes[i, j].imshow(np.squeeze(img))
                axes[i, j].axis('off')
                tittle = self.text_labels[self.train[i * num_cols + j][1]]
                if net:
                    tittle = f"{tittle}\n{self.text_labels[net(self.train[i * num_cols + j][0]).argmax(axis=1)]}"
                axes[i, j].set_title(tittle)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def time_test_dataloader(self, batch_size, num_workers):
        train, test = self.get_iter(batch_size, num_workers)
        time_start = time.time()
        for X, y in train:
            continue
        time_end = time.time()
        print(f'batch_size={batch_size}, num_workers={num_workers}, used_time={time_end - time_start:.2f} s')
