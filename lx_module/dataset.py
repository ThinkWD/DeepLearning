import os
import PIL
from tqdm import tqdm
import numpy as np
import torch
import pandas
import sklearn
import hashlib
import sklearn.model_selection
import torchvision  # 对于计算机视觉实现的一个库
import matplotlib.pyplot as plt


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def show_images(imgs, num_rows, num_cols, titles=None, scale=1, save_path=None):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def generate_dataset_preview(dataset, num2label, save_path=None, num_rows=5, num_cols=5, scale=1, net=None):
    tittles = []
    indices = range(num_rows * num_cols)
    for idx in indices:
        title = num2label[dataset[idx][1]]
        if net:
            y_hat_index = net(dataset[idx][0].to(try_gpu())).argmax(axis=1).item()
            title = f"y: {title}\ny_hat: {num2label[y_hat_index]}"
        tittles.append(title)
    imgs = [dataset[idx][0] for idx in indices]
    show_images(imgs, num_rows, num_cols, tittles, scale, save_path)


def check_dataset(root_path, csv_file):
    data_frame = pandas.read_csv(os.path.join(root_path, csv_file))
    image_set = np.asarray(data_frame.iloc[:, 0])
    label_set = np.asarray(data_frame.iloc[:, 1])
    hash_dict = {}
    duplicates = {}
    for idx, image_path in enumerate(tqdm(image_set, leave=True, ncols=100, colour="CYAN")):
        img = PIL.Image.open(os.path.join(root_path, image_path))
        img_hash = hashlib.md5(img.tobytes()).hexdigest()
        if img_hash in hash_dict:
            duplicates.setdefault(hash_dict[img_hash], []).append(idx)
        else:
            hash_dict[img_hash] = idx
    to_remove = set()
    for key, indices in duplicates.items():
        labels = [label_set[key]]
        labels.extend([label_set[idx] for idx in indices])
        if len(set(labels)) == 1:
            to_remove.update(indices[1:])  # 保留第一个，删除其余的
        else:
            to_remove.update(indices)  # 标签不同，删除所有
    data_frame = data_frame.drop(list(to_remove))
    updated_csv_file = os.path.join(root_path, f'updated_{csv_file}')
    data_frame.to_csv(updated_csv_file, index=False)
    print(f"Updated CSV file saved as {updated_csv_file}, Removed {len(to_remove)} data.")


###########################################################
#
#  数据集 自定义实现
#
###########################################################
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
    def __init__(self, true_w, true_b, train_examples, test_examples, batch_size=10, num_workers=4):
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
        batch_size = self.batch_size if batch_size <= 0 else batch_size
        num_workers = self.num_workers if num_workers <= 0 else num_workers
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
    def __init__(self, batch_size, num_workers=4, train_augs=None, test_augs=None, save_path="./dataset"):
        self.labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot']
        self.batch_size = batch_size
        self.num_workers = num_workers
        # 初始化 pipeline.
        train_augs = train_augs if train_augs else torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_augs = test_augs if test_augs else torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # 通过内置函数下载数据集到 save_path 目录下
        self.train = torchvision.datasets.FashionMNIST(root=save_path, train=True, transform=train_augs, download=True)
        self.test = torchvision.datasets.FashionMNIST(root=save_path, train=False, transform=test_augs, download=True)
        print(f'\nNumber of categories -> {len(self.labels)}')
        print(f'Original Shape -> {self.train[0][0].shape}\n')

    def generate_dataset_preview(self, save_path=None, num_rows=5, num_cols=5, scale=1, net=None):
        generate_dataset_preview(self.train, self.labels, save_path, num_rows, num_cols, scale, net)

    def get_iter(self, batch_size=0, num_workers=0):
        batch_size = self.batch_size if batch_size <= 0 else batch_size
        num_workers = self.num_workers if num_workers <= 0 else num_workers
        train = torch.utils.data.DataLoader(self.train, batch_size, shuffle=True, num_workers=num_workers)
        test = torch.utils.data.DataLoader(self.test, batch_size, shuffle=False, num_workers=num_workers)
        return train, test


# kaggle 房价预测数据集: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
class Dataset_HousePricesAdvanced(object):
    def __init__(self, batch_size=64, num_workers=4, save_path="./dataset"):
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
        self.X = torch.tensor(all_features[: train.shape[0]].values, dtype=torch.float32)
        self.y = torch.tensor(train.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
        self.test_X = torch.tensor(all_features[train.shape[0] :].values, dtype=torch.float32)

    def get_k_fold_data_iter(self, K, i, batch_size=0, num_workers=0):
        batch_size = self.batch_size if batch_size <= 0 else batch_size
        num_workers = self.num_workers if num_workers <= 0 else num_workers
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
        batch_size = self.batch_size if batch_size <= 0 else batch_size
        num_workers = self.num_workers if num_workers <= 0 else num_workers
        train_arrays = (self.X, self.y)
        train_dataset = torch.utils.data.TensorDataset(*train_arrays)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
        return train_iter


# CIFAR10 彩色分类数据集
class Dataset_CIFAR10(object):
    def __init__(self, batch_size, num_workers=4, train_augs=None, test_augs=None, save_path="./dataset"):
        self.labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.batch_size = batch_size
        self.num_workers = num_workers
        # 初始化 pipeline.
        train_augs = train_augs if train_augs else torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_augs = test_augs if test_augs else torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # 通过内置函数下载数据集到 save_path 目录下
        self.train = torchvision.datasets.CIFAR10(root=save_path, train=True, transform=train_augs, download=True)
        self.test = torchvision.datasets.CIFAR10(root=save_path, train=False, transform=test_augs, download=True)
        print(f'\nNumber of categories -> {len(self.labels)}')
        print(f'Original Shape -> {self.train[0][0].shape}\n')

    def generate_dataset_preview(self, save_path=None, num_rows=5, num_cols=5, scale=1, net=None):
        generate_dataset_preview(self.train, self.labels, save_path, num_rows, num_cols, scale, net)

    def get_iter(self, batch_size=0, num_workers=0):
        batch_size = self.batch_size if batch_size <= 0 else batch_size
        num_workers = self.num_workers if num_workers <= 0 else num_workers
        train = torch.utils.data.DataLoader(self.train, batch_size, shuffle=True, num_workers=num_workers)
        test = torch.utils.data.DataLoader(self.test, batch_size, shuffle=False, num_workers=num_workers)
        return train, test


class Custom_Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, csv_file, transform=None):
        self.root_path = root_path
        self.transform = transform if transform else torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # 图片
        data_frame = pandas.read_csv(os.path.join(self.root_path, csv_file))
        self.image_set = np.asarray(data_frame.iloc[:, 0])
        self.set_length = len(self.image_set)
        # 标签
        self.labels, self.label_set = None, None
        if 'label' in data_frame.columns:
            text_labels = np.asarray(data_frame.iloc[:, 1])  # 每张图片的文本标签
            self.labels = sorted(list(set(text_labels)))  # 总共多少个类
            text2num = {label: idx for idx, label in enumerate(self.labels)}  # 文本映射到数字
            self.label_set = np.array([text2num[label] for label in text_labels])  # 每张图片的数字标签

    def __len__(self):
        return self.set_length

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        label = self.label_set[idx] if self.label_set is not None else 0
        image = self.transform(PIL.Image.open(os.path.join(self.root_path, self.image_set[idx])))
        return image, label


# kaggle 树叶分类数据集: https://www.kaggle.com/competitions/classify-leaves/overview
class Dataset_classify_leaves(object):
    def __init__(self, batch_size, num_workers=4, train_augs=None, test_augs=None, save_path="./dataset"):
        save_path = os.path.join(save_path, 'classify-leaves')
        self.batch_size = batch_size
        self.num_workers = num_workers
        # check_dataset(save_path, 'train.csv')
        self.train = Custom_Image_Dataset(save_path, 'updated_train.csv', train_augs)
        self.test = Custom_Image_Dataset(save_path, 'updated_train.csv', test_augs)
        self.labels = self.train.get_labels()
        print(f'\nNumber of categories -> {len(self.labels)}')
        print(f'Original Shape -> {self.train[0][0].shape}, {self.train[0][1]}\n')
        self.K_fold = -1
        self.K_fold_indices = []

    def get_k_fold_data_iter(self, K_fold, i, batch_size=0, num_workers=0):
        batch_size = self.batch_size if batch_size <= 0 else batch_size
        num_workers = self.num_workers if num_workers <= 0 else num_workers
        if K_fold != self.K_fold:
            self.K_fold = K_fold
            k_fold = sklearn.model_selection.KFold(n_splits=self.K_fold, shuffle=True)
            self.K_fold_indices = list(k_fold.split(self.train))
        train_indices, test_indices = self.K_fold_indices[i]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
        train = torch.utils.data.DataLoader(self.train, batch_size, sampler=train_sampler, num_workers=num_workers)
        test = torch.utils.data.DataLoader(self.test, batch_size, sampler=test_sampler, num_workers=num_workers)
        return train, test
