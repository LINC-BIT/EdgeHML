_base_ = [
    f"./schedule_1x.py",
]

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[1200, 1600])  # 原为12万和16万
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=1800)  # 原为 18万

