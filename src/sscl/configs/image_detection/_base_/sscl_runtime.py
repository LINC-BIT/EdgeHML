_base_ = [
    f"./default_runtime.py",
]

# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,
    hooks=[
        dict(
            type="TextLoggerHook",
            by_epoch=False
        ),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="er_soft_teacher",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)

custom_hooks = [dict(type='NumClassCheckHook')]
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=20)  # 原interval为4000
