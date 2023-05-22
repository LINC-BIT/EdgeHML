mmdet_base = "../_base_"
_base_ = [
    f"{mmdet_base}/models/retinanet_r50_fpn.py",
    f"{mmdet_base}/datasets/coco_detection_sscl.py",
    f"{mmdet_base}/schedules/schedule_sscl.py",
    f"{mmdet_base}/sscl_runtime.py",
]

# ======================= 模型 ==========================：
# 基于mmdet的retinanet_free_anchor_r50_fpn_1x_coco.py

model = dict(
    type="ErFreeAnchor",
    backbone=dict(  # 根据SoftTeacher的FasterRCNN设置：
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        # style="caffe",
        # init_cfg=dict(
        #     type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        # ),
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
    bbox_head=dict(
        _delete_=True,
        type='FreeAnchorRetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.75)
    ),
    train_cfg=dict(
        buffer_size=200,
        minibatch_size=1,
    )
)

optimizer = dict(type='SGD', lr=0.000125, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(step=[3000, 4000])  # 原为12万和16万
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=5000)  # 原为 18万
evaluation = dict(type="SeqDetEvalHook", interval=1000)  # 原为4000

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
