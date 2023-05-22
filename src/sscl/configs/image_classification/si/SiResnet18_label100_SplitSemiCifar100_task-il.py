_base_ = [
    '../_base_/models/resnet18_cifar100_normal.py',
    '../_base_/datasets/split_semi_cifar100_label100_task-il.py',
    '../_base_/schedules/cifar100_bs8_sscl_1k.py',
    '../_base_/sscl_runtime.py'
]


model = dict(
    type="SiResnet18",
    train_cfg=dict(
        c=0.5,
        xi=1.0,
        lr=0.03,
    )
)

optimizer_config = dict(
    _delete_=True,
    type="SiOptimizerHook",
    grad_clip=dict(
        clip_value=1,
    ),
)

run_memo = "SI；Cifar100；迭代1k次/task"