import copy
import os

import pandas
from mmengine.config import ConfigDict
from mmengine.dist import sync_random_seed
from mmengine.fileio import dump, load
from mmengine.hooks import Hook
from mmengine.runner import Runner, find_latest_checkpoint

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
        ),
        custom_hooks=[dict(type='EMAHook', ema_type='ExponentialMovingAverage')],  # 模型参数指数滑动平均
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
    paramwise_cfg = dict()
    if freeze_param:
        paramwise_cfg = dict(
            custom_keys={
                'backbone.layer0': dict(lr_mult=0, decay_mult=0),  # 设置骨干网络第 0 层的学习率和权重衰减为 0
                'backbone': dict(lr_mult=1),  # 设置骨干网络的其余层和优化器保持一致
                'head': dict(lr_mult=10),  # 设置头部网络的学习率为优化器中设置的 10 倍
            }
        )
    cfg = dict(
        # train, val, test setting
        train_cfg=dict(by_epoch=True, max_epochs=num_epochs, val_interval=1),
        val_cfg=dict(),
        test_cfg=dict(),
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


def get_dataset_classify_leaves(config, save_path='/home/lxx/DeepLearning/dataset/classify-leaves', first_time=False):
    target_size = config['target_size']
    batch_size = config['batch_size']
    num_workers = config['num_workers'] if 'num_workers' in config else 4
    if first_time:
        # 数据集预处理: 去除重复图片, 杜绝图片相同标签不同的情况
        dataset.check_dataset(save_path, 'train.csv')
        # 数据集格式转换: 转为 CustomDataset 需要的格式
        data_frame = pandas.read_csv(os.path.join(save_path, 'updated_train.csv'))  # 加载数据集
        image_data = data_frame.iloc[:, 0]
        label_data = data_frame.iloc[:, 1]
        # 获取类别列表并按出现次数排序, 然后保存到 txt
        classes = label_data.value_counts().index.tolist()
        with open(os.path.join(save_path, 'classes.txt'), 'w', encoding='utf-8') as file:
            for cls in classes:
                file.write(f'{cls}\n')
        # 将 CSV 中的文本标签转换为标签序号, 然后按 mmpretrain 的要求保存到 txt
        label_data = label_data.map({label: idx for idx, label in enumerate(classes)})
        with open(os.path.join(save_path, 'dataset.txt'), 'w', encoding='utf-8') as file:
            for img, lbl in zip(image_data, label_data):
                file.write(f'{img} {lbl}\n')
    # 读取类别列表文件
    with open(os.path.join(save_path, 'classes.txt'), encoding='utf-8') as file:
        classes = [line.strip() for line in file]
    print(f'num_classes: {len(classes)}')
    # 设置训练和测试 pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', scale=target_size),
        dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
        dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0, backend='cv2'),
        dict(type='PackInputs'),
    ]
    test_pipeline = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=target_size), dict(type='PackInputs')]
    # 训练数据设置
    train_dataloader = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=dict(
            type='CustomDataset',
            data_root=save_path,  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
            ann_file='dataset.txt',  # 相对于 `data_root` 的标注文件路径
            classes=classes,  # 每个类别的名称
            pipeline=train_pipeline,  # 处理数据集样本的一系列变换操作
        ),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    val_dataloader = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CustomDataset',
            data_root=save_path,  # `ann_flie` 和 `data_prefix` 共同的文件路径前缀
            ann_file='dataset.txt',  # 相对于 `data_root` 的标注文件路径
            classes=classes,  # 每个类别的名称
            pipeline=test_pipeline,  # 处理数据集样本的一系列变换操作
        ),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    val_evaluator = dict(type='Accuracy', topk=(1, 5))
    return dict(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_evaluator=val_evaluator,
        test_dataloader=val_dataloader,
        test_evaluator=val_evaluator,
    )


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
            batch_size=32,
            grad_accum_steps=1,
            load_from='./checkpoints/efficientnet-b4_3rdparty-ra-noisystudent_in1k_20221103-16ba8a2d.pth',
        )
    )
    model = dict(
        type='ImageClassifier',
        backbone=dict(type='EfficientNet', arch='b4'),
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
            batch_size=32,
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

    # 更新训练集
    dataset = cfg.train_dataloader.dataset
    train_dataset = copy.deepcopy(dataset)
    cfg.train_dataloader.dataset = wrap_dataset(train_dataset, False)
    # 更新测试集、验证集加载位置及 pipeline
    if cfg.val_dataloader is not None:
        if 'pipeline' not in cfg.val_dataloader.dataset:
            raise ValueError('Cannot find `pipeline` in the val dataset. ')
        val_dataset = copy.deepcopy(dataset)
        val_dataset['pipeline'] = cfg.val_dataloader.dataset.pipeline
        cfg.val_dataloader.dataset = wrap_dataset(val_dataset, True)
    if cfg.test_dataloader is not None:
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
def train():
    resume = False
    num_splits = 5

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
        custom_cfg, cfg = get_model(num_splits, fold, True)
        cfg.update(get_schedules(custom_cfg))
        cfg.update(get_dataset_classify_leaves(custom_cfg))
        cfg.update(get_runtime(custom_cfg))
        # set the unify random seed
        cfg.kfold_split_seed = kfold_split_seed
        # train
        train_single_fold(cfg, num_splits, fold, resume_ckpt)
        resume_ckpt = None


if __name__ == '__main__':
    train()
