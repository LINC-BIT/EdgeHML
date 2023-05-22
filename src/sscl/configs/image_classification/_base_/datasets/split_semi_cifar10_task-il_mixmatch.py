_base_ = ['./pipelines/rand_aug.py']

# dataset settings
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False  # 是否反转通道，使用 cv2, mmcv 读取图片默认为 BGR 通道顺序，这里 Normalize 均值方差数组的数值是以 RGB 通道顺序， 因此需要反转通道顺序。
)

sup_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

unsup_weak_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img'])
]

unsup_strong_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img'])
]

test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])  # test 时不传递 gt_label
]

data = dict(
    samples_per_gpu=2,  # 这里表示每个 batch 包含 2 张有标签图像（和相应的无标签图像）
    workers_per_gpu=1,
    train=[
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[0, 1],
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[2, 3],
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[4, 5],
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[6, 7],
            num_label_per_class=5,
            uratio=7,
            sup_pipeline=sup_pipeline,
            unsup_weak_pipeline=unsup_weak_pipeline,
            unsup_strong_pipeline=unsup_strong_pipeline,
            test_mode=False,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[8, 9],
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
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[0, 1],
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[2, 3],
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[4, 5],
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[6, 7],
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
        dict(
            type="SplitSemiCifar10",
            root="data/cifar",
            class_index=[8, 9],
            test_pipeline=test_pipeline,
            test_mode=True,
            CL_TYPE="task-il",
        ),
    ],
    test=[],
)
