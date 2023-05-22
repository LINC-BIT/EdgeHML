_base_ = [
    f"./coco_detection.py",
]

# 原来的：
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0],
    to_rgb=False
)



# 有标签图像使用的正常pipeline：
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                # img_scale=[(600, 200), (733, 300)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="OneOf",
                transforms=[
                    dict(type=k)
                    for k in [
                        "Identity",
                        "AutoContrast",
                        "RandEqualize",
                        "RandSolarize",
                        "RandColor",
                        "RandContrast",
                        "RandBrightness",
                        "RandSharpness",
                        "RandPosterize",
                    ]
                ],
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="sup"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
        ),
    ),
]

# 无标签图像的强增强（用于学生模型学习）：
strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type=k)
                            for k in [
                                "Identity",
                                "AutoContrast",
                                "RandEqualize",
                                "RandSolarize",
                                "RandColor",
                                "RandContrast",
                                "RandBrightness",
                                "RandSharpness",
                                "RandPosterize",
                            ]
                        ],
                    ),
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", angle=(-30, 30)),
                            [
                                dict(type="RandShear", x=(-30, 30)),
                                dict(type="RandShear", y=(-30, 30)),
                            ],
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.2],
                squared=True,
            ),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

# 无标签图像的弱增强（用于教师模型计算伪标签和伪框）：
weak_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 400), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
            ),
            dict(type="RandFlip", flip_ratio=0.5),
        ],
        record=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "tag",
            "transform_matrix",
        ),
    ),
]

unsup_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations", with_bbox=True),
    # generate fake labels for data format compatibility

    # 因为无标签数据没法LoadAnnotations，
    # 所以作者为了对齐和方便后续处理，这里使用PseudoSamples存储假的标签：
    dict(type="PseudoSamples", with_bbox=True),

    dict(
        # 因为无标签图像要同时走弱增强和强增强这2条线，所以作者自己实现了MultiBranch类，
        # 这个类会在内部为弱增强标记为teacher，为强增强标记为student，
        # 并最终输出一个list。注意：因为输出的是list，所以作者实现了新的 Collate 函数来组合batch。
        # 注意：同一张图像经过弱增强和强增强之后，bbox的位置会发生改变（这一点与图像分类任务差别很大），
        # 所以需要记录两条pipeline中涉及到的所有几何变换过程，将其转换为一个变换矩阵，以计算
        # 正确的bbox位置。这一过程在GeometricTransformationBase类中实现。
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,  # 原为5，即有标签1+无标签4=5；
    workers_per_gpu=1,
    classes=("person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"),
    train=[
        dict(
            # _delete_=True,
            type="SemiDataset",  # 这个SemiDataset类就是作者自己实现的数据集类，传入sup和unsup
            sup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}/instances_train2017.${fold}@${percent}_1.json",
                img_prefix="data/coco/train2017/",
                pipeline=train_pipeline,
                filter_empty_gt=True,
            ),
            unsup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled/instances_train2017.${fold}@${percent}-unlabeled_1.json",
                img_prefix="data/coco/train2017/",
                pipeline=unsup_pipeline,
                filter_empty_gt=False,
            ),
        ),
        dict(
            # _delete_=True,
            type="SemiDataset",  # 这个SemiDataset类就是作者自己实现的数据集类，传入sup和unsup
            sup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}/instances_train2017.${fold}@${percent}_2.json",
                img_prefix="data/coco/train2017/",
                pipeline=train_pipeline,
                filter_empty_gt=True,
            ),
            unsup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled/instances_train2017.${fold}@${percent}-unlabeled_2.json",
                img_prefix="data/coco/train2017/",
                pipeline=unsup_pipeline,
                filter_empty_gt=False,
            ),
        ),
        dict(
            # _delete_=True,
            type="SemiDataset",  # 这个SemiDataset类就是作者自己实现的数据集类，传入sup和unsup
            sup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}/instances_train2017.${fold}@${percent}_3.json",
                img_prefix="data/coco/train2017/",
                pipeline=train_pipeline,
                filter_empty_gt=True,
            ),
            unsup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled/instances_train2017.${fold}@${percent}-unlabeled_3.json",
                img_prefix="data/coco/train2017/",
                pipeline=unsup_pipeline,
                filter_empty_gt=False,
            ),
        ),
        dict(
            # _delete_=True,
            type="SemiDataset",  # 这个SemiDataset类就是作者自己实现的数据集类，传入sup和unsup
            sup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}/instances_train2017.${fold}@${percent}_4.json",
                img_prefix="data/coco/train2017/",
                pipeline=train_pipeline,
                filter_empty_gt=True,
            ),
            unsup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled/instances_train2017.${fold}@${percent}-unlabeled_4.json",
                img_prefix="data/coco/train2017/",
                pipeline=unsup_pipeline,
                filter_empty_gt=False,
            ),
        ),
        dict(
            # _delete_=True,
            type="SemiDataset",  # 这个SemiDataset类就是作者自己实现的数据集类，传入sup和unsup
            sup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}/instances_train2017.${fold}@${percent}_5.json",
                img_prefix="data/coco/train2017/",
                pipeline=train_pipeline,
                filter_empty_gt=True,
            ),
            unsup=dict(
                type="CocoDataset",
                ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled/instances_train2017.${fold}@${percent}-unlabeled_5.json",
                img_prefix="data/coco/train2017/",
                pipeline=unsup_pipeline,
                filter_empty_gt=False,
            ),
        ),
    ],

    val=[
        dict(
            type="SplitCocoDataset",
            ann_file='data/coco/annotations/instances_val2017/instances_val2017_1_mini.json',
            img_prefix='data/coco/val2017/',
            pipeline=test_pipeline,
        ),
        dict(
            type="SplitCocoDataset",
            ann_file='data/coco/annotations/instances_val2017/instances_val2017_2_mini.json',
            img_prefix='data/coco/val2017/',
            pipeline=test_pipeline,
        ),
        dict(
            type="SplitCocoDataset",
            ann_file='data/coco/annotations/instances_val2017/instances_val2017_3_mini.json',
            img_prefix='data/coco/val2017/',
            pipeline=test_pipeline,
        ),
        dict(
            type="SplitCocoDataset",
            ann_file='data/coco/annotations/instances_val2017/instances_val2017_4_mini.json',
            img_prefix='data/coco/val2017/',
            pipeline=test_pipeline,
        ),
        dict(
            type="SplitCocoDataset",
            ann_file='data/coco/annotations/instances_val2017/instances_val2017_5_mini.json',
            img_prefix='data/coco/val2017/',
            pipeline=test_pipeline,
        ),
    ],

    #dict(pipeline=test_pipeline),
    test=dict(
            type="CocoDataset",
            ann_file='data/coco/annotations/instances_val2017.json',
            img_prefix='data/coco/val2017/',
            pipeline=test_pipeline,
        ),


    sampler=dict(
        train=dict(
            # 这里的 SemiBalanceSampler 最终会包装为 GroupSemiBalanceSampler，
            # “Group” 是一个目标检测领域常用于sampler的功能，会将宽高比一致的图片分组，
            # 同组内宽高比一致，组成一个 batch，防止 padding 过多。具体实现比较复杂。
            # 而这里的 SemiBalanceSampler 会实现每个 batch 中包含 1 张有标签图像，和
            # 4 张无标签图像。
            type="GroupSemiBalanceSampler",
            sample_ratio=[1, 1],  # 原为 [1,4]
            by_prob=True,
            # at_least_one=True,
            epoch_length=7330,
        )
    ),
)

