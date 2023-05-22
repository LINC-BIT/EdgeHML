# checkpoint saving
checkpoint_config = dict(
    # _delete_=True,
    by_epoch=False,
    interval=500,
    max_keep_ckpts=3
)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='SeqTensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
