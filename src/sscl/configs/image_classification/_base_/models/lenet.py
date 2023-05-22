model = dict(
    type='ImageClassifier',
    backbone=dict(type='LeNet5', num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))