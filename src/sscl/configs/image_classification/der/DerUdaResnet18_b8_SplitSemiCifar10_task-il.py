_base_ = [
    '../_base_/models/resnet18_cifar_normal.py',
    '../_base_/datasets/split_semi_cifar10_task-il.py',
    '../_base_/schedules/cifar10_bs8_sscl_1k.py',
    '../_base_/sscl_runtime.py'
]


model = dict(
    type="DerppUdaResnet18",
    train_cfg=dict(
        buffer_size=200,
        minibatch_size=8,

        alpha=0.3,  # 1.0
        beta=0,  # 0.5
        lambda_u=1.0,  # 1.0
    )
)

run_memo = "DER+UDA；迭代1k次/task"
