# model settings
"""
使用默认的 normal kaiming 初始化，而非 uniform kaiming 初始化。
"""
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=-1,  # 若num_stages=4，backbone包含stem 与 4 个 stages。frozen_stages为-1时，不冻结网络； 为0时，冻结 stem； 为1时，冻结 stem 和 stage1； 为4时，冻结整个backbone
        style='pytorch',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=200,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),  # 评估指标，Top-k 准确率， 这里为 top1 与 top5 准确率
    ))
