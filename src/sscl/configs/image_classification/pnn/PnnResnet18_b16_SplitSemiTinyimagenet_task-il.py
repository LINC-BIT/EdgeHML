_base_ = [
    '../_base_/models/resnet18_tinyimagenet_normal.py',
    '../_base_/datasets/split_semi_tinyimagenet_task-il.py',
    '../_base_/schedules/tinyimagenet_bs8_sscl_1k.py',
    '../_base_/sscl_runtime.py'
]


model = dict(
    type="PnnResnet18",
)
run_memo = "PNN；迭代1k次/task"