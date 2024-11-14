import os
import timm
import torch
import scipy
import pandas
import torchvision
from tqdm import tqdm
from lx_module import uitls
from lx_module import dataset
from timm.loss import SoftTargetCrossEntropy


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


def find_saved_model(root_path, name):
    for item in os.scandir(root_path):
        if item.is_file() and item.name.startswith(name) and item.name.endswith('.pth'):
            return item.path
    raise Exception('The model file does not exist')


def get_model(K_flod: int, flod: int, is_train: bool):
    # [trick] 模型选择: 使用多个表现较好的模型分别训练, 最后使用所有模型投票
    # [trick] batch_size 最好在 32 ~ 512 之间. 显存不足时, 用梯度累积技术模拟大 batch_size
    num_classes = 176
    if flod < K_flod:
        net = timm.create_model(  # 关掉 pretrained 并使用 print(net.default_cfg) 查看预训练模型下载链接
            'resnet50d',
            num_classes=num_classes,
            pretrained=is_train,
            pretrained_cfg_overlay=dict(file='./dataset/resnet50d_ra2-464e36ba.pth'),
        )
        net = net.to(device=uitls.try_gpu())
        logger = f'resnet50d-fold-{flod % K_flod}'
        return net, 64, 1e-4, logger
    elif flod < K_flod * 2:
        net = timm.create_model(  # 关掉 pretrained 并使用 print(net.default_cfg) 查看预训练模型下载链接
            'efficientnet_b3',
            num_classes=num_classes,
            pretrained=is_train,
            pretrained_cfg_overlay=dict(file='./dataset/efficientnet_b3_ra2-cf984f9c.pth'),
        )
        net = net.to(device=uitls.try_gpu())
        logger = f'efficientnet_b3-fold-{flod % K_flod}'
        return net, 32, 1e-4, logger
    elif flod < K_flod * 3:
        net = timm.create_model(  # 关掉 pretrained 并使用 print(net.default_cfg) 查看预训练模型下载链接
            'legacy_seresnext50_32x4d',
            num_classes=num_classes,
            pretrained=False,
            pretrained_cfg_overlay=dict(file='./dataset/legacy_se_resnext50_32x4d-f3651bad.pth'),
        )
        net = net.to(device=uitls.try_gpu())
        logger = f'seresnext50d_32x4d-fold-{flod % K_flod}'
        return net, 32, 1e-4, logger
    else:
        raise Exception('no defined')


def eval_asst(net, data_path, csv_file, transforms, save_path, delay_vote=True):
    net.eval()
    # dataset
    test_data = dataset.Custom_Image_Dataset(data_path, csv_file, transforms)
    dataloader = torch.utils.data.DataLoader(test_data, 32, shuffle=False, num_workers=4)
    # predict
    with torch.no_grad():
        preds = []
        for imgs, _ in tqdm(dataloader, leave=True, ncols=100, colour="CYAN"):
            if isinstance(imgs, list):
                imgs_size = len(imgs)
                imgs = torch.cat([img.to(uitls.try_gpu()) for img in imgs], dim=0)  # 将列表合并为一个大的batch
                y_hat = net(imgs).argmax(axis=1).view(imgs_size, -1).transpose(0, 1).cpu().numpy()
                if not delay_vote:
                    y_hat = scipy.stats.mode(y_hat, axis=1, keepdims=True)[0].reshape(-1)
            else:
                y_hat = net(imgs.to(uitls.try_gpu())).argmax(axis=1).cpu().numpy()
            preds.extend(y_hat.tolist())
    # save csv
    columns = [f'label_{i}' for i in range(len(preds[0]))] if isinstance(preds[0], list) else ['label']
    labels_df = pandas.DataFrame(preds, columns=columns, dtype=int)
    test_data = pandas.read_csv(os.path.join(data_path, csv_file))
    submission = pandas.concat([test_data['image'], labels_df], axis=1)
    submission[labels_df.columns] = submission[labels_df.columns].astype(int)
    submission.to_csv(save_path, index=False)


def eval(K_flod, target_size, checkpoints_path, save_path):
    data_path = './dataset/classify-leaves'
    os.makedirs(save_path, exist_ok=True)
    # [trick] 测试时数据增强: FiveCrop 或 TenCrop, 然后多个结果投票
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(round(target_size / 7 * 8)),  # 先缩放到一个较大的尺寸
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            torchvision.transforms.TenCrop(target_size),  # 上下左右中心裁剪, 获得 5 张图片
        ]
    )
    # 使用训练好的模型预测测试集
    for flod in range(K_flod * 3):
        net, _, _, logger = get_model(K_flod, flod, False)
        model_path = find_saved_model(checkpoints_path, logger)
        saved_path = os.path.join(save_path, f'submission_{logger}.csv')
        net.load_state_dict(torch.load(model_path))
        print(f'{uitls._time_()} - Evaluate model: \'{logger}\'')
        eval_asst(net, data_path, 'test.csv', transforms, saved_path)
        print(f'{uitls._time_()} - Done, save to: {saved_path}\n')
    # 收集全部识别结果
    result = pandas.DataFrame()
    csv_files = [item.path for item in os.scandir(save_path) if item.is_file() and item.name.endswith('.csv')]
    for idx, file in enumerate(csv_files):
        df = pandas.read_csv(file)
        label_columns = [col for col in df.columns if col.startswith('label')]
        df_labels = df[label_columns].rename(columns=lambda x: f'label_{idx}_{x}')
        result = pandas.concat([result, df_labels], axis=1)
    # 将识别结果的众数作为最终的识别结果 (存在多个众数时取第一个)
    result = result.mode(axis=1, numeric_only=True)[0].astype(int)
    num2label = dataset.Custom_Image_Dataset(data_path, 'train.csv').get_labels()  # 获取每个数字对应的文本标签
    result = result.map({k: v for k, v in enumerate(num2label)})  # 将数字标签转为文本标签
    test_data = pandas.read_csv(os.path.join(data_path, 'test.csv'))  # 获取测试集图片列表
    submission = pandas.DataFrame({'image': test_data['image'], 'label': result})  # 组合为结果文件
    submission.to_csv('./final_predictions.csv', index=False)
    print(f"{uitls._time_()} - The final result has been saved to './final_predictions.csv'")


def train(K_flod, target_size):
    # [trick] 训练时数据增强: 图片上背景较多, 且树叶没有方向性, 可以做更多增强
    train_augs = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(target_size),
            torchvision.transforms.RandomResizedCrop(target_size),  # 随机裁切
            torchvision.transforms.RandomVerticalFlip(),  # 垂直方向翻转
            torchvision.transforms.RandomHorizontalFlip(),  # 水平方向翻转
            torchvision.transforms.RandomRotation(45),  # 随机旋转 (-45° ~ 45°)
            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # hue=0.1
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 添加随机高斯噪声
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # 随机遮挡小区域
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_augs = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(target_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ### >>> 数据集 <<< ######################################################################
    # [trick] 数据集预处理: 去除重复图片, 杜绝图片相同标签不同的情况
    data = dataset.Dataset_classify_leaves(32, 8, train_augs, test_augs)
    # [trick] 训练时数据'跨图片'增强: Mixup (随机叠加两张图片) + CutMix (随机组合来自不同图片的块)
    mixup_fn = timm.data.Mixup(mixup_alpha=0.2, cutmix_alpha=0.2, num_classes=176)
    loss = SoftTargetCrossEntropy()  # torch.nn.CrossEntropyLoss()
    ### >>> 训练 <<< ########################################################################
    grad_accum_steps = 4
    total_acc = 0  # 计算K折平均精度
    for flod in range(K_flod * 3):
        # model, dataloader
        net, batch_size, learn_rate, logger = get_model(K_flod, flod, True)
        print(f'\n\nstart training {logger} with batch_size {batch_size * grad_accum_steps}')
        train_iter, test_iter = data.get_k_fold_data_iter(K_flod, flod % K_flod, batch_size)
        # [trick] 优化算法: 使用 Adam 或其变种 (避免调参)
        opt = torch.optim.AdamW(net.parameters(), learn_rate, weight_decay=1e-3)
        # [trick] 学习率策略: Warmup + Cosine
        num_epochs = 50  # 训练轮数
        warmup_epochs = 5  # 1000 iter 或 5 epoch
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-2, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        # [trick] 使用自动混合半精度加速训练过程
        total_acc += uitls.train_classification(
            net, opt, loss, train_iter, test_iter, num_epochs, logger, scheduler, grad_accum_steps, mixup_fn, True
        )
        if (flod + 1) % K_flod == 0:
            print(f'\n\n\n{logger} Average test accuracy: {total_acc / K_flod:.6f}\n\n\n')
            total_acc = 0


# 数据方面：
# - [trick] 数据集预处理: 去除重复图片, 杜绝图片相同标签不同的情况
# - [trick] 训练时数据增强: 图片上背景较多, 且树叶没有方向性, 可以做更多增强
# - [trick] 测试时数据增强: FiveCrop 或 TenCrop, 然后多个结果投票
# - [trick] 训练时数据'跨图片'增强: Mixup (随机叠加两张图片), CutMix (随机组合来自不同图片的块)
#
# 模型方面：
# - [trick] 模型选择: 使用多个表现较好的模型分别训练, 最后使用所有模型投票
# - [trick] 优化算法: 使用 Adam 或其变种 (避免调参)
# - [trick] 学习率策略: Warmup + Cosine 或 ReduceLROnPlateau 训练不动时向下调
#
# 训练方面：
# - [trick] K 折交叉验证
# - [trick] 更大的网络输入尺寸
# - [trick] batch_size 最好在 32 ~ 512 之间. 显存不足时, 用梯度累积技术模拟大 batch_size
# - [trick] 使用自动混合半精度加速训练过程

# Best Public Score: 0.97886
# Best Private Score: 0.97840
# Kaggle: https://www.kaggle.com/competitions/classify-leaves
if __name__ == "__main__":
    K_flod = 5  # [trick] K 折交叉验证
    target_size = 320  # [trick] 更大的网络输入尺寸
    train(K_flod, target_size)
    eval(K_flod, target_size, os.getcwd(), './submission')
