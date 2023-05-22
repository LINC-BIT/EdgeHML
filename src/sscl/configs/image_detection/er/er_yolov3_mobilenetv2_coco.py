mmdet_base = "../_base_"
_base_ = [
    f"{mmdet_base}/datasets/coco_detection_sscl.py",
    f"{mmdet_base}/schedules/schedule_sscl.py",
    f"{mmdet_base}/sscl_runtime.py",
]

# ======================= 模型 ==========================：
# 基于mmdet官方的yolov3_mobilenetv2_320_300e_coco.py
model = dict(
    type="ErYOLOV3",
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        norm_cfg=dict(type='BN', requires_grad=False),  # SoftTeacher
        norm_eval=True,  # SoftTeacher
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')
    ),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[320, 96, 32],
        out_channels=[96, 96, 96]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=80,
        in_channels=[96, 96, 96],
        out_channels=[96, 96, 96],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(220, 125), (128, 222), (264, 266)],
                        [(35, 87), (102, 96), (60, 170)],
                        [(10, 15), (24, 36), (72, 42)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0
        ),

        # ER need:
        buffer_size=200,
        minibatch_size=1,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100)
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
