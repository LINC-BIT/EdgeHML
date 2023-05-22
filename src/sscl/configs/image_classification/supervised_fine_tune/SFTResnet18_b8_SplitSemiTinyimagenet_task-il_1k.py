_base_ = [
    '../_base_/models/resnet18_tinyimagenet_normal.py',
    '../_base_/datasets/split_semi_tinyimagenet_task-il.py',
    '../_base_/schedules/tinyimagenet_bs8_sscl_1k.py',
    '../_base_/sscl_runtime.py'
]

model = dict(
    type="DerppResnet18",
    train_cfg=dict(
        buffer_size=200,
        minibatch_size=8,

        alpha=0,  # 1.0
        beta=0,  # 0.5
    )
)
run_memo = "SFT；迭代1k次/task"
# runner = dict(type='IterBasedRunner', max_iters=50)

# custom_hooks = [
#     dict(
#         type='EMAHook',
#         momentum=0.999,
#         warm_up=1,  # warmup为0时会报错（division by 0）
#         priority='HIGH'
#     )
# ]

# optimizer_config = dict(type="TwoStageSemiSupOptimizerHook")

# optimizer_config = dict(
#     _delete_=True,
#     grad_clip=dict(
#         max_norm=5, norm_type=2,
#     ),
# )
