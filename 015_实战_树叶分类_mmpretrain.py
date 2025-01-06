import copy
import os

import cv2
import numpy as np
import pandas
import torch
import torchvision
from mmengine.config import ConfigDict
from mmengine.dist import sync_random_seed
from mmengine.fileio import dump, load
from mmengine.hooks import Hook
from mmengine.infer import BaseInferencer
from mmengine.runner import Runner, find_latest_checkpoint
from tqdm import tqdm

from lx_module import dataset

EXP_INFO_FILE = 'kfold_exp.json'


def get_runtime(config, resume=False):
    work_name = config['work_name'] if 'work_name' in config else 'log'
    load_from = config['load_from'] if 'load_from' in config else None
    return dict(
        log_level='INFO',  # 全局日志等级
        load_from=load_from,  # 加载预训练模型
        resume=resume,  # 断点恢复训练 (设置为 True 自动从最新的权重文件恢复)
        work_dir=f'./workspace/{work_name}',  # 保存路径
        randomness=dict(seed=None, deterministic=False),  # 指定随机种子, 用于复现实验, 默认不指定.
        # 训练进程可视化 (保存到 Tensorboard 后端)
        visualizer=dict(type='UniversalVisualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
        # 设置钩子
        default_hooks=dict(
            runtime_info=dict(type='RuntimeInfoHook'),  # 往 message hub 更新运行时信息
            timer=dict(type='IterTimerHook'),  # 统计迭代耗时
            sampler_seed=dict(type='DistSamplerSeedHook'),  # 确保分布式 Sampler 的 shuffle 生效
            logger=dict(type='LoggerHook', interval=50),  # 打印日志 间隔 (iter)
            param_scheduler=dict(type='ParamSchedulerHook'),  # 启用学习率调度器
            checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto'),
            visualization=dict(type='VisualizationHook', enable=False),
        ),
        # custom_hooks=[dict(type='EMAHook', ema_type='ExponentialMovingAverage')],  # 模型参数指数滑动平均
        # 设置默认注册域 (没有此选项则无法加载 mmpretrain 中注册的类)
        default_scope='mmpretrain',
        launcher='none',
    )


def get_schedules(config):
    num_epochs = config['num_epochs']
    learn_rate = config['learn_rate']
    scheduler_type = config['scheduler_type'] if 'scheduler_type' in config else 'CosineAnnealingLR'
    grad_accum_steps = config['grad_accum_steps'] if 'grad_accum_steps' in config else 1
    freeze_param = config['freeze_param'] if 'freeze_param' in config else False
    use_amp = config['use_amp'] if 'use_amp' in config else False
    auto_scale_lr = dict(enable=False)
    if 'base_batch_size' in config:
        auto_scale_lr = dict(base_batch_size=config['base_batch_size'], enable=True)
    # 学习率调度器
    warmup = dict(type='LinearLR', start_factor=1e-4, end=round(num_epochs * 0.06), convert_to_iter_based=True)
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = dict(type='CosineAnnealingLR', by_epoch=True, convert_to_iter_based=True, T_max=num_epochs)
    elif scheduler_type == 'ReduceOnPlateauLR':
        patience = max(2, num_epochs // 10 - 1)  # factor=0.1,
        scheduler = dict(type='ReduceOnPlateauLR', monitor='accuracy/top1', rule='greater', patience=patience)
    # 是否冻结第 0 层
    paramwise_cfg = dict(norm_decay_mult=0.0)  # 防止BN层受到正则化衰减, 允许自由调整
    if freeze_param:
        paramwise_cfg = dict(
            norm_decay_mult=0.0,  # 防止BN层受到正则化衰减, 允许自由调整
            custom_keys={
                'backbone.layer0': dict(lr_mult=0, decay_mult=0),  # 设置骨干网络第 0 层的学习率和权重衰减为 0
                'backbone': dict(lr_mult=1),  # 设置骨干网络的其余层和优化器保持一致
                'head': dict(lr_mult=10),  # 设置头部网络的学习率为优化器中设置的 10 倍
            },
        )
    cfg = dict(
        # train, val, test setting
        train_cfg=dict(by_epoch=True, max_epochs=num_epochs, val_interval=1),
        val_cfg=dict(),
        # 优化器
        optim_wrapper=dict(
            optimizer=dict(type='AdamW', lr=learn_rate, weight_decay=1e-5),
            accumulative_counts=max(1, grad_accum_steps),  # 梯度累计步幅, 用于在显存不足时避免 batch_size 太小
            paramwise_cfg=paramwise_cfg,
        ),
        # 学习率调度器
        param_scheduler=[warmup, scheduler],
        # 根据实际训练批大小自动缩放lr
        auto_scale_lr=auto_scale_lr,
    )
    if use_amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
    return cfg


def get_dataset_classify_leaves(config, data_path='/home/lxx/DeepLearning/dataset/classify-leaves', first_time=False):
    target_size = config['target_size']
    batch_size = config['batch_size']
    num_workers = config['num_workers'] if 'num_workers' in config else 4
    if first_time:
        # 数据集预处理: 去除重复图片, 杜绝图片相同标签不同的情况
        dataset.check_dataset(data_path, 'train.csv')
        # 数据集格式转换: 转为 CustomDataset 需要的格式
        data_frame = pandas.read_csv(os.path.join(data_path, 'updated_train.csv'))  # 加载数据集
        image_data = data_frame.iloc[:, 0]
        label_data = data_frame.iloc[:, 1]
        # 获取类别列表并按出现次数排序, 然后保存到 txt
        classes = label_data.value_counts().index.tolist()
        with open(os.path.join(data_path, 'classes.txt'), 'w', encoding='utf-8') as file:
            for cls in classes:
                file.write(f'{cls}\n')
        # 将 CSV 中的文本标签转换为标签序号, 然后按 mmpretrain 的要求保存到 txt
        label_data = label_data.map({label: idx for idx, label in enumerate(classes)})
        with open(os.path.join(data_path, 'dataset.txt'), 'w', encoding='utf-8') as file:
            for img, lbl in zip(image_data, label_data):
                file.write(f'{img} {lbl}\n')
    # 读取类别列表文件
    with open(os.path.join(data_path, 'classes.txt'), encoding='utf-8') as file:
        classes = [line.strip() for line in file]
    print(f'num_classes: {len(classes)}')
    # 设置训练和测试 pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', scale=target_size),
        dict(type='RandomFlip', prob=[0.5], direction=['horizontal']),
        # dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
        # dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0, backend='cv2'),
        dict(type='PackInputs'),
    ]
    # 验证时使用
    val_pipeline = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=target_size), dict(type='PackInputs')]
    # 训练数据设置
    train_dataloader = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type='CustomDataset',
            data_root=data_path,  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
            ann_file='dataset.txt',  # 相对于 `data_root` 的标注文件路径
            classes=classes,  # 每个类别的名称
            pipeline=train_pipeline,  # 处理数据集样本的一系列变换操作
        ),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    val_dataloader = copy.deepcopy(train_dataloader)
    val_dataloader['dataset']['pipeline'] = val_pipeline
    val_evaluator = [
        dict(topk=(1, 5), type='Accuracy'),
        dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score']),
    ]
    return dict(train_dataloader=train_dataloader, val_dataloader=val_dataloader, val_evaluator=val_evaluator)


# 86.21 - 21.23 - tinyvit-21m_in21k-distill-pre_3rdparty_in1k-384px
def get_model_tinyvit_21m(custom_cfg, num_classes, data_preprocessor, train_cfg, loss):
    target_size = 224
    custom_cfg.update(
        dict(
            work_name=f'tinyvit_21m-{target_size}px',
            target_size=target_size,
            batch_size=32,
            grad_accum_steps=1,
            load_from='./checkpoints/tinyvit-21m_in21k-distill-pre_3rdparty_in1k-384px_20221021-65be6b3f.pth',
        )
    )
    window_size = target_size // 32
    model = dict(
        type='ImageClassifier',
        backbone=dict(
            type='TinyViT',
            arch='21m',
            img_size=(target_size, target_size),
            window_size=[window_size, window_size, window_size * 2, window_size],
            out_indices=(3,),
            drop_path_rate=0.2,
            gap_before_final_norm=True,
            init_cfg=[
                dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.0),
                dict(type='Constant', layer=['LayerNorm'], val=1.0, bias=0.0),
            ],
        ),
        head=dict(type='LinearClsHead', num_classes=num_classes, in_channels=576, loss=loss, topk=(1, 5)),
        train_cfg=train_cfg,
        data_preprocessor=data_preprocessor,
    )
    return custom_cfg, ConfigDict(model=model)


# 85.25 - 19.34 - efficientnet-b4_3rdparty-ra-noisystudent_in1k (pipeline -> EfficientNetRandomCrop)
def get_model_efficientnet_b4(custom_cfg, num_classes, data_preprocessor, train_cfg, loss):
    target_size = 224
    custom_cfg.update(
        dict(
            work_name=f'efficientnet_b4-{target_size}px',
            target_size=target_size,
            batch_size=16,  # 这个模型的 batch_size 太小, 因此在 model 中冻结了所有BN层
            grad_accum_steps=4,  # 梯度累积并不影响 BN 层, 是小 batch_size 在影响. 应用梯度累积时 batch_size 一般较小
            load_from='./checkpoints/efficientnet-b4_3rdparty-ra-noisystudent_in1k_20221103-16ba8a2d.pth',
        )
    )
    model = dict(
        type='ImageClassifier',
        backbone=dict(type='EfficientNet', arch='b4', norm_cfg=dict(type='BN', requires_grad=False)),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(type='LinearClsHead', num_classes=num_classes, in_channels=1792, loss=loss, topk=(1, 5)),
        train_cfg=train_cfg,
        data_preprocessor=data_preprocessor,
    )
    return custom_cfg, ConfigDict(model=model)


# 83.67 - 18.51 - edgenext-base_3rdparty-usi_in1k
def get_model_edgenext_base(custom_cfg, num_classes, data_preprocessor, train_cfg, loss):
    target_size = 224
    custom_cfg.update(
        dict(
            work_name=f'edgenext_base-{target_size}px',
            target_size=target_size,
            batch_size=64,
            grad_accum_steps=1,
            load_from='./checkpoints/edgenext-base_3rdparty-usi_in1k_20220801-909e8939.pth',
        )
    )
    model = dict(
        type='ImageClassifier',
        backbone=dict(
            type='EdgeNeXt',
            arch='base',
            out_indices=(3,),
            drop_path_rate=0.1,
            gap_before_final_norm=True,
            init_cfg=[
                dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.0),
                dict(type='Constant', layer=['LayerNorm'], val=1.0, bias=0.0),
            ],
        ),
        head=dict(type='LinearClsHead', num_classes=num_classes, in_channels=584, loss=loss, topk=(1, 5)),
        train_cfg=train_cfg,
        data_preprocessor=data_preprocessor,
    )
    return custom_cfg, ConfigDict(model=model)


# 83.36 - 15.62 - convnext-v2-nano_fcmae-in21k-pre_3rdparty_in1k-384px
def get_model_convnext_v2_nano(custom_cfg, num_classes, data_preprocessor, train_cfg, loss):
    target_size = 224
    custom_cfg.update(
        dict(
            work_name=f'convnext_v2_nano-{target_size}px',
            target_size=target_size,
            batch_size=64,
            grad_accum_steps=1,
            load_from='./checkpoints/convnext-v2-nano_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-f951ae87.pth',
        )
    )
    model = dict(
        type='ImageClassifier',
        backbone=dict(type='ConvNeXt', arch='nano', drop_path_rate=0.1, layer_scale_init_value=0.0, use_grn=True),
        head=dict(type='LinearClsHead', num_classes=num_classes, in_channels=640, loss=loss, topk=(1, 5)),
        train_cfg=train_cfg,
        data_preprocessor=data_preprocessor,
        init_cfg=dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=0.02, bias=0.0),
    )
    return custom_cfg, ConfigDict(model=model)


def get_model(K_flod: int, flod: int, use_mixup: bool = False):
    custom_cfg = dict(
        num_epochs=50,
        learn_rate=1e-4,
        scheduler_type='ReduceOnPlateauLR',  # 'ReduceOnPlateauLR', CosineAnnealingLR
        freeze_param=False,
    )
    num_classes = 176
    data_preprocessor = dict(
        num_classes=num_classes,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],  # [123.675, 116.28, 103.53]
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],  # [58.395, 57.12, 57.375]
        to_rgb=True,
    )
    # 是否启用 mixup 和 cutmix
    train_cfg = dict()
    loss = dict(type='CrossEntropyLoss', loss_weight=1.0)
    if use_mixup:
        train_cfg = dict(augments=[dict(type='Mixup', alpha=0.8), dict(type='CutMix', alpha=1.0)], probs=[0.3, 0.7])
        loss = dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, num_classes=num_classes, reduction='mean', loss_weight=1.0
        )
    if flod < K_flod:
        return get_model_tinyvit_21m(custom_cfg, num_classes, data_preprocessor, train_cfg, loss)
    if flod < K_flod * 2:
        return get_model_efficientnet_b4(custom_cfg, num_classes, data_preprocessor, train_cfg, loss)
    if flod < K_flod * 3:
        return get_model_edgenext_base(custom_cfg, num_classes, data_preprocessor, train_cfg, loss)
    if flod < K_flod * 4:
        return get_model_convnext_v2_nano(custom_cfg, num_classes, data_preprocessor, train_cfg, loss)
    else:
        raise Exception('no defined')


def train_single_fold(cfg, num_splits, fold, resume_ckpt=None):
    root_dir = cfg.work_dir
    cfg.work_dir = os.path.join(root_dir, f'fold{fold}')
    if resume_ckpt is not None:
        cfg.resume = True
        cfg.load_from = resume_ckpt

    def wrap_dataset(dataset, test_mode):
        return dict(
            type='KFoldDataset',
            dataset=dataset,
            fold=fold,
            num_splits=num_splits,
            seed=cfg.kfold_split_seed,
            test_mode=test_mode,
        )

    # 保留训练集的 pipeline, 然后将 dataset 更新为 KFoldDataset
    dataset = cfg.train_dataloader.dataset
    train_dataset = copy.deepcopy(dataset)
    cfg.train_dataloader.dataset = wrap_dataset(train_dataset, False)
    # 保留测试集、验证集的 pipeline, 然后将 dataset 更新为 KFoldDataset
    if cfg.val_dataloader is not None:
        if 'pipeline' not in cfg.val_dataloader.dataset:
            raise ValueError('Cannot find `pipeline` in the val dataset. ')
        val_dataset = copy.deepcopy(dataset)
        val_dataset['pipeline'] = cfg.val_dataloader.dataset.pipeline
        cfg.val_dataloader.dataset = wrap_dataset(val_dataset, True)
    if 'test_dataloader' in cfg and cfg.test_dataloader is not None:
        if 'pipeline' not in cfg.test_dataloader.dataset:
            raise ValueError('Cannot find `pipeline` in the test dataset. ')
        test_dataset = copy.deepcopy(dataset)
        test_dataset['pipeline'] = cfg.test_dataloader.dataset.pipeline
        cfg.test_dataloader.dataset = wrap_dataset(test_dataset, True)

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.logger.info(f'----------- Cross-validation: [{fold+1}/{num_splits}] -----------')
    runner.logger.info(f'Train dataset: \n{runner.train_dataloader.dataset}')

    class SaveInfoHook(Hook):
        def after_train_epoch(self, runner):
            last_ckpt = find_latest_checkpoint(cfg.work_dir)
            exp_info = dict(fold=fold, last_ckpt=last_ckpt, kfold_split_seed=cfg.kfold_split_seed)
            dump(exp_info, os.path.join(root_dir, EXP_INFO_FILE))

    runner.register_hook(SaveInfoHook(), 'LOWEST')
    # start training
    runner.train()


# 查看可视化曲线: tensorboard --logdir=<directory_name>
def train(num_splits: int = 5):
    resume = False

    # resume from the previous experiment
    if resume:
        experiment_info = load(os.path.join('./workspace/ResNeSt50_32xb64-256px', EXP_INFO_FILE))
        resume_fold = experiment_info['fold']
        resume_ckpt = experiment_info.get('last_ckpt', None)
        kfold_split_seed = experiment_info['kfold_split_seed']
    else:
        resume_fold = 0
        resume_ckpt = None
        kfold_split_seed = sync_random_seed()

    for fold in range(resume_fold, num_splits * 4):
        # build train config
        custom_cfg, cfg = get_model(num_splits, fold, False)
        cfg.update(get_schedules(custom_cfg))
        cfg.update(get_dataset_classify_leaves(custom_cfg))
        cfg.update(get_runtime(custom_cfg))
        # set the unify random seed
        cfg.kfold_split_seed = kfold_split_seed
        # train
        train_single_fold(cfg, num_splits, fold % num_splits, resume_ckpt)
        resume_ckpt = None


def find_best_checkpoint(path: str):
    best_checkpoint = [
        item.path
        for item in os.scandir(path)
        if item.is_file() and item.name.startswith('best_') and item.name.endswith('.pth')
    ]
    assert len(best_checkpoint) > 0, f'Failed to find the best model on {path}'
    return best_checkpoint[0]


class CustomInferencer(BaseInferencer):
    def _init_pipeline(self, cfg):
        from mmengine.dataset import Compose
        from mmpretrain.registry import TRANSFORMS

        def load_image(input_):
            img = cv2.imread(input_)
            if img is None:
                raise ValueError(f'Failed to read image {input_}.')
            return dict(img=img, img_shape=img.shape[:2], ori_shape=img.shape[:2])

        if True:  # 使用 torchvision 的 pipeline
            pipeline = [dict(type='PackInputs')]
            pipeline = Compose([TRANSFORMS.build(t) for t in pipeline])
        else:  # 使用配置文件中的 pipeline
            pipe_cfg = cfg.val_dataloader.dataset.pipeline
            assert pipe_cfg[0].type == 'LoadImageFromFile'
            pipeline = Compose([TRANSFORMS.build(t) for t in pipe_cfg[1:]])
            pipeline = Compose([load_image, pipeline])
        return pipeline

    def visualize(self, inputs=None, preds=None, show=None):
        pass

    def postprocess(self, preds, visualization, return_datasamples=False):  # noqa: F811
        if return_datasamples:
            return preds
        result = []
        for data_sample in preds:
            label = torch.argmax(data_sample.pred_score).item()
            result.append(label)
        return result


# PIL 转 tensor (opencv 格式)
def mmengine_to_tensor(img):
    img = torch.from_numpy(np.array(img, copy=True))
    img = img.permute((2, 0, 1)).contiguous()  # HWC to CHW
    img = img.flip(0)  # RGB to BGR
    # print(f'[load] shape: {list(img.shape)}, mean: {torch.mean(img, dim=(1, 2), dtype=float).tolist()}')
    return img


# tensor 转 mmengine 推理需要的格式
def tensor_to_mmegine(imgs):
    shape = tuple(imgs[0].shape[-2:])  # imgs 是一个 batch 的数据
    if isinstance(imgs, list):
        imgs = torch.cat(imgs, dim=0)
    return [dict(img=img, img_shape=shape, ori_shape=shape) for img in imgs]


# 投票刷精度
def assign_votes(datasamples, crop_size, topk=10):
    assert topk > 0
    datalength = len(datasamples)
    assert datalength % crop_size == 0
    num_image = datalength // crop_size
    labels = torch.full((num_image, topk * crop_size), -1, dtype=int, device=datasamples[0].pred_score.device)
    threshold = 1 - 0.5 / topk
    for idx in range(datalength):
        score, label = torch.topk(datasamples[idx].pred_score, topk)
        row = idx % num_image
        col = idx // num_image * topk
        if score[0] > threshold:
            labels[row, col : col + topk] = label[0]
        else:
            index = col
            total_score = score.sum()
            for j in range(1, topk):
                count = round(topk * (score[j] / total_score).item())
                if count == 0:
                    break
                labels[row, index : index + count] = label[j]
                index += count
            labels[row, index : col + topk] = label[0]
    return labels.cpu().numpy().tolist()


def save_csv_preds(images, preds, save_path):
    columns = [f'{i}' for i in range(len(preds[0]))] if isinstance(preds[0], list) else ['label']
    labels_df = pandas.DataFrame(preds, columns=columns, dtype=int)
    submission = pandas.concat([images, labels_df], axis=1)
    submission.to_csv(save_path, index=False)
    print(f'Done, save to: {save_path}\n')


def collect_submission(csv_dir, num2label, images):
    csv_files = [item.path for item in os.scandir(csv_dir) if item.is_file() and item.name.endswith('.csv')]
    result = pandas.DataFrame()
    for idx, file in enumerate(csv_files):
        df = pandas.read_csv(file)
        df_labels = df[df.columns[1:]].rename(columns=lambda x: f'l_{idx}_{x}')
        result = pandas.concat([result, df_labels], axis=1)
    result = result.mode(axis=1, numeric_only=True)[0].astype(int)  # 对多个结果取众数 (存在多个众数时取第一个)
    result = result.map(num2label)  # 将数字标签转为文本标签
    submission = pandas.DataFrame({'image': images, 'label': result})
    save_path = os.path.join(csv_dir, 'final_predictions.csv')
    submission.to_csv(save_path, index=False)
    print(f"The final result has been saved to '{save_path}'")


def test(num_splits: int = 5):
    from mmengine import init_default_scope

    # dataset path
    data_path = './dataset/classify-leaves'
    save_dir_one = './workspace/submission_topk_one'
    save_dir_ten = './workspace/submission_topk_ten'
    os.makedirs(save_dir_one, exist_ok=True)
    os.makedirs(save_dir_ten, exist_ok=True)
    test_images = pandas.read_csv(os.path.join(data_path, 'test.csv'))['image']
    # dataset
    target_size = 224
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Lambda(lambda x: mmengine_to_tensor(x)),  # PIL 转 tensor (opencv 格式)
            torchvision.transforms.Resize(round(target_size / 7 * 8)),  # 缩放到一个较大的尺寸
            torchvision.transforms.TenCrop(target_size),  # 上下左右中心裁剪+翻转, 获得 10 张图片
        ]
    )
    test_data = dataset.Custom_Image_Dataset(data_path, 'test.csv', transforms)
    dataloader = torch.utils.data.DataLoader(test_data, 16, shuffle=False, num_workers=4)
    # 调用所有模型生成识别结果投票
    for fold in range(num_splits * 4):
        # build train config
        custom_cfg, cfg = get_model(num_splits, fold, False)
        fold = fold % num_splits
        print(f"==================== {custom_cfg['work_name']}-fold{fold} ====================")
        cfg.update(get_schedules(custom_cfg))
        cfg.update(get_dataset_classify_leaves(custom_cfg))
        cfg.update(get_runtime(custom_cfg))
        # init_default_scope
        init_default_scope(cfg.default_scope)
        # get this fold best checkpoints
        cfg.work_dir = os.path.join(cfg.work_dir, f'fold{fold}')
        cfg.load_from = find_best_checkpoint(cfg.work_dir)
        # build model and Inferencer
        topk_one = []
        topk_ten = []
        inferencer = CustomInferencer(model=cfg, weights=cfg.load_from, show_progress=False)
        for imgs, _ in tqdm(dataloader, leave=True, ncols=100, colour='CYAN'):
            datasamples = inferencer(tensor_to_mmegine(imgs), batch_size=16, return_datasamples=True)
            topk_one.extend(assign_votes(datasamples, len(imgs) if isinstance(imgs, list) else 1, 1))
            topk_ten.extend(assign_votes(datasamples, len(imgs) if isinstance(imgs, list) else 1, 10))
        # save csv
        save_csv_preds(test_images, topk_one, os.path.join(save_dir_one, f"{custom_cfg['work_name']}-fold{fold}.csv"))
        save_csv_preds(test_images, topk_ten, os.path.join(save_dir_ten, f"{custom_cfg['work_name']}-fold{fold}.csv"))
    # 收集投票并生成最终结果
    print('==================== final_predictions ====================')
    with open(os.path.join(data_path, 'classes.txt'), encoding='utf-8') as file:
        num2label = {idx: line.strip() for idx, line in enumerate(file)}
    collect_submission(save_dir_one, num2label, test_images)
    collect_submission(save_dir_ten, num2label, test_images)


if __name__ == '__main__':
    test()
    # train()
