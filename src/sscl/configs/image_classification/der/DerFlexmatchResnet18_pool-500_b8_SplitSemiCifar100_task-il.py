_base_ = [
    '../_base_/models/resnet18_cifar100_normal.py',
    '../_base_/datasets/split_semi_cifar100_task-il.py',
    '../_base_/schedules/cifar100_bs8_sscl_1k.py',
    '../_base_/sscl_runtime.py'
]


model = dict(
    type="DerppFlexmatchResnet18",
    train_cfg=dict(
        buffer_size=500,
        minibatch_size=8,

        alpha=0.3,  # 1.0
        beta=0,  # 0.5
        lambda_u=1.0,  # 1.0
    )
)

run_memo = "DER+Flexmatch；Cifar100;迭代1k次/task"