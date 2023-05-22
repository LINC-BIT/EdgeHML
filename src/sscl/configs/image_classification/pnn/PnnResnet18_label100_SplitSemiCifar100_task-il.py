_base_ = [
    '../_base_/models/resnet18_cifar100_normal.py',
    '../_base_/datasets/split_semi_cifar100_label100_task-il.py',
    '../_base_/schedules/cifar100_bs8_sscl_1k.py',
    '../_base_/sscl_runtime.py'
]


model = dict(
    type="PnnResnet18",
)
run_memo = "PNN；Cifar100;迭代1k次/task"