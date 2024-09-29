import time
import torch
import pandas
import numpy as np
import torchvision  # 对于计算机视觉实现的一个库
import matplotlib.pyplot as plt


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


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


def get_k_fold_data(K, i, X, y):
    '''用于 K 折交叉验证, 获取 K 折的第 i 个切片'''
    assert K > 1
    fold_size = X.shape[0] // K
    X_train, y_train = None, None
    for j in range(K):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


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
        # MasVnrType 特征有 5 种取值：BrkCmn; BrkFace; NA; None; Stone. 其中 NA 表示的是缺失值, None 表示的是没有外部砌体
        # 但 pandas 会把 NA 和 None 都视为缺失值, 这导致 None 类型被错误理解为缺失值.
        # 因此需要在读取 CSV 文件时, 使用 keep_default_na=False 和 na_values 参数来明确指定哪些值应该被视为缺失值.
        train = pandas.read_csv(f'{save_path}/HousePricesAdvanced/train.csv', keep_default_na=False, na_values=['NA'])
        # 只有 train 部分带有标签, 可以训练. test 部分没有标签, 无法参与训练, 只能用于提交结果
        self.test = pandas.read_csv(
            f'{save_path}/HousePricesAdvanced/test.csv', keep_default_na=False, na_values=['NA']
        )
        # print(train.shape)
        # print(train.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

        ### >>> 数据预处理(数值类) <<< ##########################################################################
        # 由于我们可以提前拿到用于提交结果的 test 数据, 因此在数据标准化时可以直接考虑所有数据, 这样在 test 中结果会更好
        # 这是一种取巧的办法, 实际的应用场景中不会有这种事情, 但是为了打比赛有更好的结果, 还是必须得做.
        # 将 train 和 test 打包 (不参与标准化的数据: 第一列的 ID, 和训练集最后一列的标签)
        all_features = pandas.concat((train.iloc[:, 1:-1], self.test.iloc[:, 1:]))
        # 提取数值类数据的索引 (只有数值类数据参与标准化)
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        # 数据标准化 (将所有数值特征的均值变成0方差变成1) (每个数值都减去这一列的均值然后除以这一列的方差)
        all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
        # 将缺失值置为均值 (在标准化数据之后，所有均值变成0了)
        all_features[numeric_features] = all_features[numeric_features].fillna(0)

        ### >>> 数据预处理(非数值类) <<< ##########################################################################
        # 对于非数值类数据, 使用独热编码替换它们 (dummy_na=True 将缺失值也视为一个单独的类)
        # get_dummies 可能会将类别编码为 boolean 类型, 它无法被转为 tensor. 因此需要指定转为 int 类型
        all_features = pandas.get_dummies(all_features, dummy_na=True, dtype=int)
        # print(all_features.shape)

        ### >>> 得到数据和标签 <<< ##########################################################################
        # 关于训练集和测试集的划分, 可以简单的按比例划分, 但为了打比赛, 还是使用 K 折交叉验证
        self.X = torch.tensor(all_features[: train.shape[0]].values, dtype=torch.float32, device=try_gpu())
        self.y = torch.tensor(train.SalePrice.values.reshape(-1, 1), dtype=torch.float32, device=try_gpu())
        self.test_X = torch.tensor(all_features[train.shape[0] :].values, dtype=torch.float32, device=try_gpu())

    def get_k_fold_data_iter(self, K, i, batch_size=0, num_workers=0):
        if batch_size <= 0:
            batch_size = self.batch_size
        if num_workers <= 0:
            num_workers = self.num_workers
        train_X, train_y, test_X, test_y = get_k_fold_data(K, i, self.X, self.y)
        train_arrays = (train_X, train_y)
        train_dataset = torch.utils.data.TensorDataset(*train_arrays)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        test_arrays = (test_X, test_y)
        test_dataset = torch.utils.data.TensorDataset(*test_arrays)
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)
        return train_iter, test_iter

    def get_test_data(self):
        return self.test, self.test_X

    def get_train_data_iter(self, batch_size=0, num_workers=0):
        if batch_size <= 0:
            batch_size = self.batch_size
        if num_workers <= 0:
            num_workers = self.num_workers
        train_arrays = (self.X, self.y)
        train_dataset = torch.utils.data.TensorDataset(*train_arrays)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        return train_iter
