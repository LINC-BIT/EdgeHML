_base_ = [
    '../_base_/schedules/mnist_bs16_sscl.py',
    '../_base_/sscl_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='LeNet5', num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
# dataset settings
dataset_type = 'MNIST'
img_norm_cfg = dict(mean=[33.46], std=[78.87], to_rgb=True)
train_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=32),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=[dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=train_pipeline)],
    val=[dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=test_pipeline)],
    test=[dict(
        type=dataset_type, data_prefix='data/mnist', pipeline=test_pipeline)])

