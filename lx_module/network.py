import torch


###########################################################
#
#  特殊通用结构 自定义实现
#
###########################################################
def softmax(X):
    """softmax 函数
    1. 对每个项求幂 (使用exp)
    2. 对每一行求和得到每个样本的规范化常数
    3. 将每一行除以其规范化常数, 确保结果的和为 1
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # (广播机制)


def relu(X):
    """激活函数 relu, 就是 max(x, 0)"""
    return torch.max(X, torch.zeros_like(X))


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


class net_linear_regression_custom(object):
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
    init_weights = lambda m: torch.nn.init.normal_(m.weight, std=0.01) if isinstance(m, torch.nn.Linear) else None
    net.apply(init_weights)
    return net


class net_softmax_regression_custom(object):
    def __init__(self, num_inputs, num_outputs):
        """网络结构: Softmax 回归 的自定义实现
        Args:
            num_inputs (int): 输入特征向量的长度, 决定权重参数数量
            num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
        """
        # 权重 w 使用高斯分布(均值0方差0.01) 初始化为随机值, 偏差 b 初始化为 0
        self.w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, X):
        X = X.reshape((-1, self.w.shape[0]))  # 前处理: 将原始图像(三维)展平为向量(一维)
        X = torch.matmul(X, self.w) + self.b  # 输出层: Linear 全连接层
        return softmax(X)  # 后处理: softmax 函数将预测值转为属于每个类的概率


def net_multilayer_perceptrons(num_inputs, num_outputs, num_hiddens):
    """网络结构: 多层感知机
    Args:
        num_inputs (int): 输入特征向量的长度, 决定权重参数数量
        num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
        num_hiddens (list): 超参数. 隐藏层的数量和每层的大小
    """
    # 前处理: 将原始图像(三维)展平为向量(一维)
    layers = [torch.nn.Flatten()]
    # 创建隐藏层
    last_num_inputs = num_inputs
    for num_hidden in num_hiddens:
        layers.append(torch.nn.Linear(last_num_inputs, num_hidden))  # 隐藏层: Linear 全连接层
        layers.append(torch.nn.ReLU())  # 隐藏层的激活函数
        last_num_inputs = num_hidden
    # 创建输出层. (后处理 softmax 没有被显式定义是因为 CrossEntropyLoss 中已经包含了 softmax, 不需要重复定义)
    layers.append(torch.nn.Linear(last_num_inputs, num_outputs))
    # 创建 torch.nn 模型结构
    net = torch.nn.Sequential(*layers)
    # 参数初始化函数(lambda): 当 m 是 torch.nn.Linear 类型时初始化其权重, 否则什么也不做
    init_weights = lambda m: torch.nn.init.normal_(m.weight, std=0.01) if isinstance(m, torch.nn.Linear) else None
    net.apply(init_weights)
    return net


class net_multilayer_perceptrons_custom(object):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        """网络结构: 多层感知机 的自定义实现
        Args:
            num_inputs (int): 输入特征向量的长度, 决定权重参数数量
            num_outputs (int): 输出向量的长度, 即类别总数, 决定输出维度和偏移参数数量
            num_hiddens (list): 超参数. 隐藏层的数量和每层的大小
        """
        self.params = []
        # 创建隐藏层
        last_num_inputs = num_inputs
        for num_hidden in num_hiddens:
            self.params.append(torch.normal(0, 0.01, size=(last_num_inputs, num_hidden), requires_grad=True))
            self.params.append(torch.zeros(num_hidden, requires_grad=True))
            last_num_inputs = num_hidden
        # 创建输出层
        self.params.append(torch.normal(0, 0.01, size=(last_num_inputs, num_outputs), requires_grad=True))
        self.params.append(torch.zeros(num_outputs, requires_grad=True))

    def parameters(self):
        return self.params

    def __call__(self, X):
        # 前处理: 将原始图像(三维)展平为向量(一维)
        X = X.reshape((-1, self.params[0].shape[0]))
        # 隐藏层: 全连接层, 逐层应用权重、偏置和激活函数
        for i in range(0, len(self.params) - 2, 2):
            X = relu(torch.matmul(X, self.params[i]) + self.params[i + 1])
        # 输出层: 全连接层, 应用权重、偏置
        X = torch.matmul(X, self.params[-2]) + self.params[-1]
        # 后处理: softmax 函数将预测值转为属于每个类的概率
        return softmax(X)
