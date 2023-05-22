# optimizer
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='step', step=[2000])
lr_config = dict(policy='step', step=[800, 900])
# lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.003)
runner = dict(type='IterBasedRunner', max_iters=1000)

evaluation = dict(type="SeqClsEvalHook", interval=100)  # 原为4000