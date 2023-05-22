mmdet_base = "../_base_"
_base_ = [
    f"{mmdet_base}/datasets/coco_detection_sscl.py",
    f"{mmdet_base}/schedules/schedule_sscl.py",
    f"{mmdet_base}/sscl_runtime.py",
]

# ======================= 模型 ==========================：

model = dict(
    type='ErSSD',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(4, 7),
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03, requires_grad=False),  # SoftTeacher
        norm_eval=True,  # SoftTeacher
        # init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')
    ),
    neck=dict(
        type='SSDNeck',
        in_channels=(96, 1280),
        out_channels=(96, 1280, 512, 256, 256, 128),
        level_strides=(2, 2, 2, 2),
        level_paddings=(1, 1, 1, 1),
        l2_norm_scale=None,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(96, 1280, 512, 256, 256, 128),
        num_classes=80,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),

        # set anchor size manually instead of using the predefined
        # SSD300 setting.
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            strides=[16, 32, 64, 107, 160, 320],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            min_sizes=[48, 100, 150, 202, 253, 304],
            max_sizes=[100, 150, 202, 253, 304, 320]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,

        buffer_size=200,
        minibatch_size=1,
    ),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200)
)

optimizer = dict(type='SGD', lr=0.000125, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(step=[3000, 4000])  # 原为12万和16万
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=5000)  # 原为 18万
evaluation = dict(type="SeqEvalHook", interval=1000)  # 原为4000

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
