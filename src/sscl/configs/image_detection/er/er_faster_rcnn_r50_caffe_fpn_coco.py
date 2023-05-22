mmdet_base = "../_base_"
_base_ = [
    f"{mmdet_base}/models/faster_rcnn_r50_fpn.py",
    f"{mmdet_base}/datasets/coco_detection_sscl.py",
    f"{mmdet_base}/schedules/schedule_sscl.py",
    f"{mmdet_base}/sscl_runtime.py",
]

# ======================= 模型 ==========================：
model = dict(
    type="ErFasterRCNN",
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"
        ),
    ),
    train_cfg=dict(
        buffer_size=200,
        minibatch_size=1,
    )
)

# 注意：这里的学习率我是按照“有标签图像”的数量来设定的，
# 例如，mmdet 官方的 batch_size 为 2，而我的有标签 batch_size 为 1，
# 所以学习率为 0.02 / 2 = 0.01
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# evaluation = dict(type="SeqEvalHook", interval=500)  # 原为4000

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
