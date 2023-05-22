_base_ = ['./pipelines/rand_aug.py']

# dataset settings

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

sup_pipeline = [
    dict(type='Resize', size=64),
    dict(type='RandomCrop', size=64, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

unsup_weak_pipeline = [
    dict(type='Resize', size=64),
    dict(type='RandomCrop', size=64, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img'])
]

unsup_strong_pipeline = [
    dict(type='Resize', size=64),
    dict(type='RandomCrop', size=64, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        # 参考：mmcls\datasets\pipelines\auto_augment.py
        # 注意：这里的候选项与torchssl中的RandAugment不完全一样
        type='RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=3,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(type='Cutout', shape=5),  # 在torchssl库中，这里的cutout是随机大小的，针对FixMatch实现
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img'])
]

test_pipeline = [
    dict(type='Resize', size=64),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])  # test 时不传递 gt_label
]

data = dict(
    samples_per_gpu=8,  # 这里表示每个 batch 包含 2 张有标签图像（和相应的无标签图像）
    workers_per_gpu=1,
    train=[
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=0,
            class_index={
                "begin": 0,
                "end": 20,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=1,
            class_index={
                "begin": 20,
                "end": 40,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=2,
            class_index={
                "begin": 40,
                "end": 60,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=3,
            class_index={
                "begin": 60,
                "end": 80,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=4,
            class_index={
                "begin": 80,
                "end": 100,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=5,
            class_index={
                "begin": 100,
                "end": 120,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=6,
            class_index={
                "begin": 120,
                "end": 140,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=7,
            class_index={
                "begin": 140,
                "end": 160,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=8,
            class_index={
                "begin": 160,
                "end": 180,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=9,
            class_index={
                "begin": 180,
                "end": 200,
            },
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
    ],
    val=[
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=0,
            class_index={
                "begin": 0,
                "end": 20,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=1,
            class_index={
                "begin": 20,
                "end": 40,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=2,
            class_index={
                "begin": 40,
                "end": 60,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=3,
            class_index={
                "begin": 60,
                "end": 80,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=4,
            class_index={
                "begin": 80,
                "end": 100,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=5,
            class_index={
                "begin": 100,
                "end": 120,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=6,
            class_index={
                "begin": 120,
                "end": 140,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=7,
            class_index={
                "begin": 140,
                "end": 160,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=8,
            class_index={
                "begin": 160,
                "end": 180,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiTinyimagenet",
            root="data/tinyimagenet",
            task_id=9,
            class_index={
                "begin": 180,
                "end": 200,
            },
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
    ],
    test=[],
)
