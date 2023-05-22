_base_ = [
    '../_base_/models/resnet18_cifar_normal.py',
    '../_base_/datasets/split_semi_cifar10_task-il.py',
    '../_base_/schedules/cifar10_bs8_sscl_1k.py',
    '../_base_/sscl_runtime.py'
]

"""
本文所提方法
"""

model = dict(
    type="EdgehmlResnet18_v10",
    train_cfg=dict(
        pool_sizes=(200, 2000),
        minibatch_size=8,
        minibatch_size_unsup=56,

        lambda_sup=1,
        lambda_sup_replay=1,
        lambda_unsup=1,
        lambda_unsup_replay=0.1,

        need_sup_replay=True,
        need_unsup_replay=True,

        unsup_iter_threshold=200,
        unsup_replay_iter_threshold=-1,
        sup_iter_threshold=-1,

        threshold_length=0,
        include_harddisk_read_time=False,
    )
)
run_memo = "v10"