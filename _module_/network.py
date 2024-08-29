import torch


###########################################################
#
#  网络结构 自定义实现
#
###########################################################
def net_linear_regression(num_key_factors):
    """网络结构: 线性回归模型
    Args:
        num_key_factors (int): 影响模型结果的关键因素的数量
    """
    net = torch.nn.Sequential(torch.nn.Linear(num_key_factors, 1))
    net[0].weight.data.normal_(0, 0.01)  # w
    net[0].bias.data.fill_(0)  # b
    return net


class net_linear_regression_custom(object):
    def __init__(self, num_key_factors):
        """网络结构: 线性回归模型 的自定义实现
        Args:
            num_key_factors (int): 影响模型结果的关键因素的数量
        """
        self.w = torch.normal(0, 0.01, size=(num_key_factors, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def parameters(self):
        return self.w, self.b

    def __call__(self, X):
        return torch.matmul(X, self.w) + self.b


def softmax(X):
    """softmax 函数
    1. 对每个项求幂 (使用exp)
    2. 对每一行求和得到每个样本的规范化常数
    3. 将每一行除以其规范化常数, 确保结果的和为 1
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # (广播机制)


class net_softmax_regression_custom(object):
    def __init__(self, img_width, img_height, num_classes):
        """网络结构: Softmax 回归. 解决分类问题. 初始化网络参数函数

        注意事项:
            此处默认图片是灰度图片
            我们的输入图片的形状是 (1, w, h), 它是一个三维的输入。
            但是对于 Softmax 回归来讲, 输入需要是一个向量(一维).
            所以我们需要将这个图片展平拉长, 拉成一个长度为 1*w*h 的向量.
            当然拉成向量会失去图片本身的空间信息, 这个问题会在卷积神经网络章节继续讨论.

        Args:
            img_width (int): 输入图片宽度, 决定权重参数数量
            img_height (int): 输入图片高度, 决定权重参数数量
            num_classes (int): 类别总数, 决定输出维度和偏移参数数量
        """
        self.w = torch.normal(0, 0.01, size=(img_width * img_height, num_classes), requires_grad=True)
        self.b = torch.zeros(num_classes, requires_grad=True)

    def parameters(self):
        return [self.w, self.b]

    def __call__(self, X):
        temp = X.reshape((-1, self.w.shape[0]))  # 将原始图像(三维)展平为向量(一维)
        temp = torch.matmul(temp, self.w) + self.b  # 全连接计算
        return softmax(temp)  # softmax 计算
