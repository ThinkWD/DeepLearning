import os
import pandas
from lx_module import dataset
from mmengine.config import ConfigDict
from mmengine.runner import Runner, find_latest_checkpoint
from mmengine.hooks import Hook
from mmengine.fileio import dump, load


import copy
from mmengine.dist import sync_random_seed

EXP_INFO_FILE = 'kfold_exp.json'


# '/home/lxx/mmpretrain/configs/resnet/resnet18_8xb32_in1k.py'


def get_runtime(work_name='log', load_from=None, resume=False):
    return dict(
        # 指定随机种子, 用于复现实验, 默认不指定.
        randomness=dict(seed=None, deterministic=False),
        # 初始化参数模型路径
        load_from=load_from,
        # 断点恢复训练 (设置为 True 自动从最新的权重文件恢复)
        resume=resume,
        # 全局日志等级
        log_level='INFO',
        # 保存路径
        work_dir=f'./workspace/{work_name}',
        # 训练进程可视化 (保存到 Tensorboard 后端)
        # visualizer=dict(type='UniversalVisualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
        visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')]),
        # 设置默认钩子
        default_hooks=dict(
            # 保存权重的间隔 (epoch)
            checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto'),
            logger=dict(type='LoggerHook', interval=50),  # 打印日志的间隔 (iter)
            timer=dict(type='IterTimerHook'),  # 记录每次迭代的时间
            param_scheduler=dict(type='ParamSchedulerHook'),  # 启用学习率调度器
            sampler_seed=dict(type='DistSamplerSeedHook'),  # 在分布式环境中设置采样器种子
            visualization=dict(type='VisualizationHook', enable=False),  # 验证结果可视化，设置True以启用它。
        ),
        # 设置默认注册域 (没有此选项则无法加载 mmpretrain 中注册的类)
        default_scope='mmpretrain',
        launcher='none',
    )


def get_schedules(
    learn_rate, num_epochs, scheduler_type='CosineAnnealingLR', grad_accum_steps=1, freeze_param=False, use_amp=False
):
    # 学习率调度器
    if scheduler_type == 'CosineAnnealingLR':
        scheduler = dict(type='CosineAnnealingLR', by_epoch=True)  # , T_max=num_epochs
    elif scheduler_type == 'ReduceOnPlateauLR':
        scheduler = dict(type='ReduceOnPlateauLR', by_epoch=True, patience=4, factor=0.2, monitor='accuracy/top1')
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
            optimizer=dict(type='AdamW', lr=learn_rate, weight_decay=1e-4),
            accumulative_counts=max(1, grad_accum_steps),  # 梯度累计步幅, 用于在显存不足时避免 batch_size 太小
            paramwise_cfg=paramwise_cfg,
        ),
        # 学习率调度器
        param_scheduler=[dict(type='LinearLR', start_factor=1e-3, by_epoch=False, end=1000), scheduler],
        # 根据实际训练批大小自动缩放lr
        auto_scale_lr=dict(base_batch_size=256, enable=False),
    )
    if use_amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')
    return cfg


def get_dataset_classify_leaves(
    target_size, batch_size, num_workers=4, save_path='/home/lxx/DeepLearning/dataset/classify-leaves', first_time=False
):
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
                file.write(f"{cls}\n")
        # 将 CSV 中的文本标签转换为标签序号, 然后按 mmpretrain 的要求保存到 txt
        label_data = label_data.map({label: idx for idx, label in enumerate(classes)})
        with open(os.path.join(save_path, 'dataset.txt'), 'w', encoding='utf-8') as file:
            for img, lbl in zip(image_data, label_data):
                file.write(f"{img} {lbl}\n")
    # 读取类别列表文件
    with open(os.path.join(save_path, 'classes.txt'), 'r', encoding='utf-8') as file:
        classes = [line.strip() for line in file]
    print(f'num_classes: {len(classes)}')
    # 设置训练和测试 pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', scale=target_size),
        dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
        dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        dict(type='PackInputs'),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=target_size),
        dict(type='PackInputs'),
    ]
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


def get_model(num_classes, use_mixup=False):
    # 是否启用 mixup 和 cutmix
    train_cfg = dict()
    if use_mixup:
        train_cfg = dict(
            augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0),
            ],
            probs=[0.2, 0.2],
        )
    model = dict(
        type='ImageClassifier',
        backbone=dict(type='ResNet', depth=18, num_stages=4, out_indices=(3,), style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=num_classes,
            in_channels=512,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
        ),
        data_preprocessor=dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        train_cfg=train_cfg,
    )
    return dict(model=model)


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
            exp_info = dict(
                fold=fold,
                last_ckpt=last_ckpt,
                kfold_split_seed=cfg.kfold_split_seed,
            )
            dump(exp_info, os.path.join(root_dir, EXP_INFO_FILE))

    runner.register_hook(SaveInfoHook(), 'LOWEST')
    # start training
    runner.train()


def main():

    num_splits = 5
    resume = False

    cfg = ConfigDict()
    cfg.update(get_model(176))
    cfg.update(get_runtime())
    cfg.update(get_schedules(1e-3, 50, 'ReduceOnPlateauLR'))
    cfg.update(get_dataset_classify_leaves(224, 64))

    # set the unify random seed
    cfg.kfold_split_seed = sync_random_seed()

    # resume from the previous experiment
    if resume:
        experiment_info = load(os.path.join(cfg.work_dir, EXP_INFO_FILE))
        resume_fold = experiment_info['fold']
        cfg.kfold_split_seed = experiment_info['kfold_split_seed']
        resume_ckpt = experiment_info.get('last_ckpt', None)
    else:
        resume_fold = 0
        resume_ckpt = None

    for fold in range(resume_fold, num_splits):
        cfg_ = copy.deepcopy(cfg)
        train_single_fold(cfg_, num_splits, fold, resume_ckpt)
        resume_ckpt = None


if __name__ == '__main__':
    main()
