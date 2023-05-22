#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

import copy
import torch
from torch.nn import functional as F
from mmcls.models import CLASSIFIERS
from mmcls.models.classifiers.image import ImageClassifier
from alipy.query_strategy.query_labels import QueryInstanceUncertainty, QueryInstanceQBC

from sscl.utils.buffer import Buffer
from sscl.models import BaseLearningMethod
from ssl_utils.models.softteacher.ssod.utils import get_root_logger

import datetime

@CLASSIFIERS.register_module(force=True)
class DerppResnet18(ImageClassifier, BaseLearningMethod):
    ALG_NAME = "DerppResnet18"  # 算法模型的名称
    ALG_COMPATIBILITY = ["class-il", "task-il", "domain-il"]

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(DerppResnet18, self).__init__(
                 backbone,
                 neck=neck,
                 head=head,
                 pretrained=pretrained,
                 train_cfg=train_cfg,
                 init_cfg=init_cfg
        )
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

        self.alpha = train_cfg.alpha
        self.beta = train_cfg.beta

        self.learn_timer = 0
        self.learn_timer_per_task = 0
        self.learn_task_cnt = 0

        self.learn_iter_cnt = 0
        self.learn_timer_per_iter = 0
        self.timer100_min = float('inf')

        self.logger = get_root_logger()

        # 主动学习
        if hasattr(train_cfg, "active_learning") and len(str(train_cfg.active_learning)) > 0:
            self.active_learning = train_cfg.active_learning
        else:
            self.active_learning = None


    def before_task(self, dataloaders, runner, *args, **kwargs):
        self.cur_task_class_index = dataloaders[0].dataset.class_index
        self.num_class_per_task = len(self.cur_task_class_index)

        if self.active_learning is not None:
            # 使用指定的主动学习算法重新采样标注样本
            self.logger.info(f"开始使用 {self.active_learning} 主动学习算法重新采样标注样本")
            ds = dataloaders[0].dataset
            candidates = {}
            for idx, img_info in enumerate(ds.unlabeled_data_infos):
                lab = img_info['gt_label'].item()
                if lab not in candidates:
                    candidates[lab] = []
                candidates[lab].append(
                    dict(
                        **(ds.sup_pipeline(
                            copy.deepcopy(img_info)
                        )),
                        global_id=idx
                    )
                )
            new_data_infos = []
            for gt_label, cands in candidates.items():
                features = torch.stack([x['img'] for x in candidates[gt_label]]).cuda()
                predicts = self.get_logits(features).detach().cpu().numpy()
                features = features.detach().cpu().numpy()
                # 基于不确定性的方法：
                if self.active_learning == "entropy" or self.active_learning == "least_confident" or self.active_learning == "margin":
                    # 参考：http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/huangsj/alipy/page_reference/api_classes/api_query_strategy.query_labels.QueryInstanceUncertainty.html
                    features = features.reshape(len(features), -1)
                    al = QueryInstanceUncertainty(
                        features,
                        [0] * len(features),
                        measure=self.active_learning
                    )
                    selected = al.select_by_prediction_mat(
                        [x['global_id'] for x in candidates[gt_label]],
                        predicts,
                        batch_size=ds.num_label_per_class
                    )
                # 基于委员会：
                elif self.active_learning == "qbc":
                    # 参考：http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/huangsj/alipy/page_reference/api_classes/api_query_strategy.query_labels.QueryInstanceUncertainty.html
                    al = QueryInstanceQBC(
                        features
                        [0] * len(features)
                    )
                    selected = al.select_by_prediction_mat(
                        [x['global_id'] for x in candidates[gt_label]],
                        predicts,
                        batch_size=ds.num_label_per_class
                    )
                # 基于多样性的方法：
                # elif self.active_learning == "graph_density":
                #     # 参考：http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/huangsj/alipy/page_reference/api_classes/api_query_strategy.query_labels.QueryInstanceUncertainty.html
                #     al = QueryInstanceGraphDensity(
                #         features
                #         [0] * len(features)
                #     )
                #     selected = al.select_by_prediction_mat(
                #         [x['global_id'] for x in candidates[gt_label]],
                #         predicts,
                #         batch_size=ds.num_label_per_class
                #     )
                # 基于期望误差：
                # elif self.active_learning == "lal":
                #     # 参考：http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/huangsj/alipy/page_reference/api_classes/api_query_strategy.query_labels.QueryInstanceLAL.html
                #     features = features.reshape(len(features), -1)
                #     al = QueryInstanceLAL(
                #         features,
                #         [0] * len(features),
                #         data_path='./data/lal_train_data',
                #     )
                #     al.train_selector_from_file()
                #     selected = al.select(
                #         [],
                #         [x['global_id'] for x in candidates[gt_label]],
                #         batch_size=ds.num_label_per_class
                #     )
                else:
                    raise NotImplementedError
                for idx in selected:
                    new_data_infos.append(
                        copy.deepcopy(ds.unlabeled_data_infos[idx])
                    )
                self.logger.info(f"标注样本重新采样结果：类别 {gt_label}：{selected}")

            ds.data_infos = new_data_infos

    def get_logits(self, img):
        x = self.extract_feat(img)
        x = self.head.pre_logits(x)
        cls_score = self.head.fc(x)
        return cls_score

    def sup_forward_train(self, img, gt_label, **kwargs):
        """
        该方法来自 mmcls\models\classifiers\image.py 中 ImageClassifier，
        因为需要额外返回 logits，所以重写该方法
        :param img:
        :param gt_label:
        :param kwargs:
        :return:
        """

        losses = {}

        """
        为了提取 logits，需要分解 loss = self.head.forward_train(x, gt_label)，
        参考 mmcls\models\heads\linear_head.py 中的 LinearClsHead
        """
        cls_score = self.get_logits(img)
        loss = self.head.loss(cls_score, gt_label)
        losses['sup_loss'] = loss['loss']

        return losses, cls_score

    def derpp_forward_train(self, img, gt_label, logits: torch.Tensor, **kwargs):
        loss = {}

        assert self.alpha >= 0 and self.beta >= 0

        # buffer中的data replay：
        if not self.buffer.is_empty():  # 从buffer中选取logits和labels用于replay
            if self.alpha > 0:
                buf_inputs, _, buf_logits = self.buffer.get_data(
                    self.minibatch_size,
                    transform=None
                )
                buf_inputs = buf_inputs.to(img.device)
                buf_logits = buf_logits.to(img.device)

                buf_outputs = self.get_logits(buf_inputs)

                alpha_loss = self.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss['alpha_loss'] = alpha_loss

            if self.beta > 0:
                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.minibatch_size,
                    transform=None
                )
                buf_inputs = buf_inputs.to(img.device)
                buf_labels = buf_labels.to(img.device)

                buf_outputs = self.get_logits(buf_inputs)

                beta_loss = self.beta * F.cross_entropy(buf_outputs, buf_labels)
                loss['beta_loss'] = beta_loss

        if self.alpha > 0 or self.beta > 0:
            logits_leaved_graph = logits.detach()  # 这里需要detach，不然会报graph相关错误
            self.buffer.add_data(
                examples=img,  # 在原DERPP官方实现中，这里保存的应该是没有被transform的img
                labels=gt_label,
                logits=logits_leaved_graph
            )

        return loss

    def after_task(self, *args, **kwargs):
        pass

    def forward_train(self, img, gt_label, **kwargs):
        """

        :param img: (N, C, H, W)
        :param gt_label: (N, 1)
        :param kwargs: 包含了 img_metas 和 unsup 样本数据，但当前方法没有用到
                kwargs['unsup']['weak'][0] = {
                    'img_metas': [img_meta_dict, img_meta_dict, ...],
                    'img': N*C*H*W 的 tensor
        :return:
        """

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        losses = {}
        losses_sup, logits = self.sup_forward_train(img, gt_label, **kwargs)
        losses.update(losses_sup)
        losses_derpp = self.derpp_forward_train(img, gt_label, logits, **kwargs)
        losses.update(losses_derpp)


        return losses

# ==============================================================
# 以下是完全使用 sscl/backbone/image_classification_pytorch 中的 CNN 作为整个网络，

@CLASSIFIERS.register_module(force=True)
class DerppMobilenetv2(DerppResnet18):
    ALG_NAME = "DerppMobilenetv2"
    SKIP_TEST_BEFORE_START = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.net = None

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.mobilenet import mobilenetv2
        return mobilenetv2(num_classes)

    def to(self, *args, **kwargs):
        if self.net is not None:
            self.net.to(*args, **kwargs)

    def before_task(self, dataloaders, runner, *args, **kwargs):
        if self.net is None:
            dl = dataloaders[0]
            num_classes = len(dl.dataset.CLASSES)
            self.net = self.get_backbone(num_classes)

            self.net.to(runner.optimizer.param_groups[0]['params'][0].device)
            # 直接删除optimizer中的原param_group然后添加新的param_group会报错，
            # 因为optimizer内部还存储有原来param_group的参数，所以会导致optimzer.step
            # 时找不到原param_group。所以只能重新创建optimizer再绑定到runner上了：
            obj_cls = type(runner.optimizer)
            optimizer_config = copy.deepcopy(runner.optimizer.defaults)
            optimizer_config['params'] = self.net.parameters()
            del runner.optimizer
            runner.optimizer = obj_cls(**optimizer_config)
            print(f"已更换 optimizer：对新的 net 使用 defaults config 构造了 optimizer")

    def simple_test(self, img, img_metas=None, **kwargs):
        cls_score = self.net(img)
        softmax = kwargs['softmax'] if 'softmax' in kwargs else True
        post_process = kwargs['post_process'] if 'post_process' in kwargs else True

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.head.post_process(pred)
        else:
            return pred

    def get_logits(self, img):
        x = self.net(img)
        return x


@CLASSIFIERS.register_module(force=True)
class DerppRAN(DerppMobilenetv2):
    ALG_NAME = "DerppRAN"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.residual_attention_network import ResidualAttentionModel_92_32input_update as RAN
        return RAN(num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppCBAM(DerppMobilenetv2):
    ALG_NAME = "DerppCBAM"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.cbam import cbamnet
        return cbamnet(network_type='CIFAR100', num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppResnext29(DerppMobilenetv2):
    ALG_NAME = "DerppResnext29"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.resnext import resnext29_2x64d
        return resnext29_2x64d(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppInceptionv3(DerppMobilenetv2):
    ALG_NAME = "DerppInceptionv3"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.inception import inceptionv3
        return inceptionv3(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppSeresnet18(DerppMobilenetv2):
    ALG_NAME = "DerppSeresnet18"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.senet import senet18
        return senet18(num_classes=num_classes)


@CLASSIFIERS.register_module(force=True)
class DerppVGG16(DerppMobilenetv2):
    ALG_NAME = "DerppVGG16"

    def get_backbone(self, num_classes):
        from sscl.backbone.image_classification_pytorch.vgg import vgg16
        return vgg16(num_classes=num_classes)
