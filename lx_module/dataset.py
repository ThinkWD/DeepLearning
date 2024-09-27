import time
import torch
import pandas
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


def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise"""
    # 生成形状为 (num_examples, len(w)) 的矩阵, 矩阵用均值为 0 方差为 1 的随机数来填充
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # y = Xw + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    y = torch.reshape(y, (-1, 1))  # 转为列向量
    return X, y


class Dataset_GaussianDistribution(object):
    def __init__(self, true_w, true_b, train_examples, test_examples, batch_size=10, num_workers=8):
        """高斯分布数据集 (y = X * w + b + 噪声)
        Returns:
            X (矩阵, 形状为[num_examples, len(weight)]): 生成的数据.
            y (向量, 长度为 num_examples): 生成的数据对应的标签.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_X, self.train_y = synthetic_data(true_w, true_b, train_examples)
        self.test_X, self.test_y = synthetic_data(true_w, true_b, test_examples)

    def get_iter(self, batch_size=0, num_workers=0):
        if batch_size <= 0:
            batch_size = self.batch_size
        if num_workers <= 0:
            num_workers = self.num_workers
        train_arrays = (self.train_X, self.train_y)
        train_dataset = torch.utils.data.TensorDataset(*train_arrays)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        test_arrays = (self.test_X, self.test_y)
        test_dataset = torch.utils.data.TensorDataset(*test_arrays)
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)
        return train_iter, test_iter

    def gen_preview_image(self, save_path=None):
        plt.scatter(self.train_X[:, 1].detach().numpy(), self.train_y.detach().numpy(), 1)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# 一个简单的服装分类数据集
class Dataset_FashionMNIST(object):
    def __init__(self, batch_size=64, num_workers=8, pipeline=defult_pipeline, save_path="./dataset"):
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        # 初始化 pipeline.
        transforms = torchvision.transforms.Compose(pipeline)
        # 通过内置函数下载数据集到 save_path 目录下
        self.train = torchvision.datasets.FashionMNIST(root=save_path, train=True, transform=transforms, download=True)
        self.test = torchvision.datasets.FashionMNIST(root=save_path, train=False, transform=transforms, download=True)

    def get_iter(self, batch_size=0, num_workers=0):
        if batch_size <= 0:
            batch_size = self.batch_size
        if num_workers <= 0:
            num_workers = self.num_workers
        train = torch.utils.data.DataLoader(self.train, batch_size, shuffle=True, num_workers=num_workers)
        test = torch.utils.data.DataLoader(self.test, batch_size, shuffle=False, num_workers=num_workers)
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


# kaggle 房价预测数据集: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
class Dataset_HousePricesAdvanced(object):
    def __init__(self, batch_size=64, num_workers=8, save_path="./dataset"):
        self.batch_size = batch_size
        self.num_workers = num_workers

        ### >>> 从文件中加载数据集 <<< ##########################################################################
        train_data = pandas.read_csv(f'{save_path}/HousePricesAdvanced/train.csv')
        test_data = pandas.read_csv(f'{save_path}/HousePricesAdvanced/test.csv')
        # print(train_data.shape)
        # print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

        ### >>> 数据打包 <<< ##########################################################################
        # 训练集去掉第一列的 ID 和最后一列的标签(成交房价) - 测试集不包括标签 所以只去掉第一列的 ID 即可
        # 将训练集和测试集打包
        # 如果训练时没有测试数据, 可以只在训练集上计算均值和方差, 然后应用到测试数据中
        all_features = pandas.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

        ### >>> 数据预处理(数值类) <<< ##########################################################################
        # 提取数值类数据的索引 (排除文本类数据)
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        # 数据标准化 (将所有数值特征的均值变成0方差变成1) (每个数值都减去这一列的均值然后除以这一列的方差)
        all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
        # 将缺失值置为均值 (在标准化数据之后，所有均值变成0了)
        all_features[numeric_features] = all_features[numeric_features].fillna(0)

        ### >>> 数据预处理(非数值类) <<< ##########################################################################
        # # 编码之前: 查看哪些列是文本类数据, 以及这些列有多少个类别
        # for object in all_features.dtypes[all_features.dtypes == 'object'].index:
        #     print(object.ljust(20), len(all_features[object].unique())) # ljust 使打印数据更美观

        # 对于文本类数据, 使用独热编码替换它们 (dummy_na=True 将缺失值也视为一个单独的类)
        all_features = pandas.get_dummies(all_features, dummy_na=True)
        all_features = all_features.astype(np.float32)  # 确保所有列都是数值类型

        n_train = train_data.shape[0]
        self.train_X = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
        self.train_y = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
        self.test_X = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
        self.test_y = torch.zeros((self.test_X.shape[0], 1))  # 数据集中没有 test 部分的标签, 为了保持一致性全部设为 0

    def get_iter(self, batch_size=0, num_workers=0):
        if batch_size <= 0:
            batch_size = self.batch_size
        if num_workers <= 0:
            num_workers = self.num_workers
        train_arrays = (self.train_X, self.train_y)
        train_dataset = torch.utils.data.TensorDataset(*train_arrays)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        test_arrays = (self.test_X, self.test_y)
        test_dataset = torch.utils.data.TensorDataset(*test_arrays)
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)
        return train_iter, test_iter
