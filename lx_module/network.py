import torch


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


###########################################################
#
#  特殊通用结构 自定义实现
#
###########################################################
def relu(X):
    """激活函数 relu, 就是 max(x, 0)"""
    return torch.max(X, torch.zeros_like(X))


class BaseModel:
    def __init__(self):
        self.is_train = False

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False


###########################################################
#
#  网络结构 自定义实现
#
###########################################################
def net_linear_regression(num_key_factors, generator=0.01):
    """网络结构: 线性回归模型
    Args:
        num_key_factors (int): 影响模型结果的关键因素的数量
        generator (float): 初始化参数使用的方差 (均值默认为 0)
    """
    net = torch.nn.Sequential(torch.nn.Linear(num_key_factors, 1))
    net[0].weight.data.normal_(0, generator)  # w
    net[0].bias.data.fill_(0)  # b
    return net


class net_linear_regression_custom(BaseModel):
    def __init__(self, num_key_factors, generator=0.01):
        """网络结构: 线性回归模型 的自定义实现
        Args:
            num_key_factors (int): 影响模型结果的关键因素的数量
            generator (float): 初始化参数使用的方差 (均值默认为 0)
        """
        self.w = torch.normal(0, generator, size=(num_key_factors, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def parameters(self):
        return self.w, self.b

    def __call__(self, X):
        return torch.matmul(X, self.w) + self.b  # 没有前后处理, 只有一层输出层


def net_softmax_regression(num_inputs, num_outputs):
    """网络结构: softmax 回归
    Args:
        num_inputs (int): 输入特征向量的长度, 决定权重参数数量
        num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
    """
    net = torch.nn.Sequential(
        torch.nn.Flatten(),  # 前处理: 将原始图像(三维)展平为向量(一维)
        torch.nn.Linear(num_inputs, num_outputs),  # 输出层: Linear 全连接层
        # 后处理 softmax 没有被显式定义是因为 CrossEntropyLoss 中已经包含了 softmax, 不需要重复定义
    )
    # 参数初始化函数(lambda): 当 m 是 torch.nn.Linear 类型时初始化其权重, 否则什么也不做
    init_weights = lambda m: torch.nn.init.xavier_normal_(m.weight) if isinstance(m, torch.nn.Linear) else None
    net.apply(init_weights)
    return net


class net_softmax_regression_custom(BaseModel):
    def __init__(self, num_inputs, num_outputs):
        """网络结构: Softmax 回归 的自定义实现
        Args:
            num_inputs (int): 输入特征向量的长度, 决定权重参数数量
            num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
        """
        # 权重 w 使用高斯分布(均值0方差0.01) 初始化为随机值, 偏差 b 初始化为 0
        variance = 2 / (num_inputs + num_outputs)
        self.w = torch.normal(0, variance, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, X):
        X = X.reshape((-1, self.w.shape[0]))  # 前处理: 将原始图像(三维)展平为向量(一维)
        X = torch.matmul(X, self.w) + self.b  # 输出层: Linear 全连接层
        return X  # 后处理: softmax 函数将预测值转为属于每个类的概率 (定义在损失函数中)


def net_multilayer_perceptrons(num_inputs, num_outputs, num_hiddens, dropout=[]):
    """网络结构: 多层感知机
    Args:
        num_inputs (int): 输入特征向量的长度, 决定权重参数数量
        num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
        num_hiddens (list): 超参数. 隐藏层的数量和每层的大小
    """
    # 确保 dropout 数组与隐藏层数量一致
    if len(dropout) < len(num_hiddens):
        dropout = dropout + [0.0] * (len(num_hiddens) - len(dropout))
    else:
        dropout = dropout[: len(num_hiddens)]
    # 前处理: 将原始图像(三维)展平为向量(一维)
    layers = [torch.nn.Flatten()]
    # 创建隐藏层
    last_num_inputs = num_inputs
    for i, num_hidden in enumerate(num_hiddens):
        layers.append(torch.nn.Linear(last_num_inputs, num_hidden))  # 隐藏层: Linear 全连接层
        layers.append(torch.nn.ReLU())  # 隐藏层的激活函数
        if 0 < dropout[i] <= 1:
            layers.append(torch.nn.Dropout(dropout[i]))  # 应用 dropout
        last_num_inputs = num_hidden
    # 创建输出层. (后处理 softmax 没有被显式定义是因为 CrossEntropyLoss 中已经包含了 softmax, 不需要重复定义)
    layers.append(torch.nn.Linear(last_num_inputs, num_outputs))
    # 创建 torch.nn 模型结构
    net = torch.nn.Sequential(*layers)
    # 参数初始化函数(lambda): 当 m 是 torch.nn.Linear 类型时初始化其权重, 否则什么也不做
    init_weights = lambda m: torch.nn.init.xavier_normal_(m.weight) if isinstance(m, torch.nn.Linear) else None
    net.apply(init_weights)
    return net.to(device=try_gpu())


def dropout_layer(X, dropout, is_train=False):
    '''以 dropout 的概率随机丢弃输入 X 中的元素'''
    assert 0 <= dropout <= 1
    if dropout == 0 or is_train == False:
        return X
    if dropout == 1:
        return torch.zeros_like(X)
    # torch.rand 生成 0 ~ 1 之间的均匀随机分布, 大于 dropout 部分置1, 小于的部分置零, 得到 mask
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    # 在最后除以 1 - p 是为了保持输出的期望值不变。
    # 随机丢弃一部分神经元的输出会使得剩余的神经元输出变得稀疏。
    # 如果不进行调整，剩余神经元的输出总和会变小，从而影响模型的训练效果。
    return mask * X / (1.0 - dropout)


class net_multilayer_perceptrons_custom(BaseModel):
    def __init__(self, num_inputs, num_outputs, num_hiddens, dropout=[]):
        """网络结构: 多层感知机 的自定义实现
        Args:
            num_inputs (int): 输入特征向量的长度, 决定权重参数数量
            num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
            num_hiddens (list): 超参数. 隐藏层的数量和每层的大小
            dropout (list): (丢弃法插件) 每层的 dropout 概率
        """
        self.params = []
        # 确保 dropout 数组与隐藏层数量一致
        if len(dropout) < len(num_hiddens):
            self.dropout = dropout + [0.0] * (len(num_hiddens) - len(dropout))
        else:
            self.dropout = dropout[: len(num_hiddens)]
        # 创建隐藏层
        last_num_inputs = num_inputs
        for num_hidden in num_hiddens:
            size = (last_num_inputs, num_hidden)
            variance = 2 / (last_num_inputs + num_hidden)
            self.params.append(torch.normal(0, variance, size=size, requires_grad=True, device=try_gpu()))
            self.params.append(torch.zeros(num_hidden, requires_grad=True, device=try_gpu()))
            last_num_inputs = num_hidden
        # 创建输出层
        size = (last_num_inputs, num_outputs)
        variance = 2 / (last_num_inputs + num_outputs)
        self.params.append(torch.normal(0, variance, size=size, requires_grad=True, device=try_gpu()))
        self.params.append(torch.zeros(num_outputs, requires_grad=True, device=try_gpu()))

    def parameters(self):
        return self.params

    def __call__(self, X):
        # 前处理: 将原始图像(三维)展平为向量(一维)
        X = X.reshape((-1, self.params[0].shape[0]))
        # 隐藏层: 全连接层, 逐层应用权重、偏置、激活函数和丢弃法
        for i in range(0, len(self.params) - 2, 2):
            X = relu(torch.matmul(X, self.params[i]) + self.params[i + 1])  # 全连接层计算+激活函数
            X = dropout_layer(X, self.dropout[i // 2], self.is_train)  # 应用丢弃法
        # 输出层: 全连接层, 应用权重、偏置
        X = torch.matmul(X, self.params[-2]) + self.params[-1]
        return X  # 后处理: softmax 函数将预测值转为属于每个类的概率 (定义在损失函数中)
