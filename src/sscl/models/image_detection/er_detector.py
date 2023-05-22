import torch

from sscl.utils.buffer import Buffer
from sscl.models.utils.base_model import BaseLearningMethod

from mmdet.models import DETECTORS
from mmdet.models.detectors import FasterRCNN, YOLOV3, RetinaNet, SingleStageDetector, DETR
from ssl_utils.models.softteacher.ssod.utils import log_every_n
from ssl_utils.models.softteacher.ssod.utils.structure_utils import dict_split


@DETECTORS.register_module(force=True)
class ErFasterRCNN(FasterRCNN, BaseLearningMethod):
    ALG_NAME = 'er_faster_rcnn'
    ALG_COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck,
                 pretrained,
                 init_cfg)

        # 因为需要支持mmdet的多卡训练，所以buffer先存在cpu上，需要计算时再放到指定cuda上：
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        if gt_bboxes_ignore is not None or gt_masks is not None or proposals is not None:
            raise NotImplementedError

        # 过滤出有标签数据：
        sup_img = None
        sup_img_metas = []
        sup_gt_bboxes = []
        sup_gt_labels = []
        for img_idx, img_meta in enumerate(img_metas):
            if img_meta['tag'] == 'sup':
                if sup_img is None:
                    sup_img = img[img_idx].unsqueeze(0)
                else:
                    sup_img = torch.cat([sup_img, img[img_idx].unsqueeze(0)])
                sup_img_metas.append(img_meta)
                sup_gt_bboxes.append(gt_bboxes[img_idx])
                sup_gt_labels.append(gt_labels[img_idx])

        loss = super().forward_train(
                      sup_img,
                      sup_img_metas,
                      sup_gt_bboxes,
                      sup_gt_labels,
                      gt_bboxes_ignore=None,
                      sup_gt_masks=None,
                      sup_proposals=None,
                      **kwargs)

        buf_data_size = None
        if not self.buffer.is_empty():
            # 这里的返回值顺序可以参考 self.buffer.attributes
            buf_img, buf_img_matas, buf_gt_bboxes, buf_gt_labels = self.buffer.get_data(
                self.minibatch_size, transform=None  # 在目标检测任务中，向buffer中存的就是经过transform的图像，所以此时无需transform
            )
            buf_data_size = buf_img.shape[0]

            buf_img = buf_img.to(img.device)
            buf_gt_labels = [x.to(gt_labels[0].device) for x in buf_gt_labels]
            buf_gt_bboxes = [x.to(gt_bboxes[0].device) for x in buf_gt_bboxes]

            er_loss = super().forward_train(
                buf_img,
                buf_img_matas,
                buf_gt_bboxes,
                buf_gt_labels,
                gt_bboxes_ignore=None,
                gt_masks=None,
                proposals=None,
                **kwargs
            )
            er_loss = {"er_" + k: v for k, v in er_loss.items()}
            loss.update(**er_loss)

        self.buffer.add_data(
            examples=sup_img,
            img_metas=sup_img_metas,
            gt_bboxes=sup_gt_bboxes,
            gt_labels=sup_gt_labels,
        )

        return loss


@DETECTORS.register_module(force=True)
class ErYOLOV3(YOLOV3, BaseLearningMethod):
    ALG_NAME = 'er_yolov3'
    ALG_COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)

        # 因为需要支持mmdet的多卡训练，所以buffer先存在cpu上，需要计算时再放到指定cuda上：
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        if gt_bboxes_ignore is not None:
            raise NotImplementedError

        # 过滤出有标签数据：
        sup_img = None
        sup_img_metas = []
        sup_gt_bboxes = []
        sup_gt_labels = []
        # 过滤出有标签数据：
        for img_idx, img_meta in enumerate(img_metas):
            if img_meta['tag'] == 'sup':
                if sup_img is None:
                    sup_img = img[img_idx].unsqueeze(0)
                else:
                    sup_img = torch.cat([sup_img, img[img_idx].unsqueeze(0)])
                sup_img_metas.append(img_meta)
                sup_gt_bboxes.append(gt_bboxes[img_idx])
                sup_gt_labels.append(gt_labels[img_idx])


        loss = super().forward_train(
                      sup_img,
                      sup_img_metas,
                      sup_gt_bboxes,
                      sup_gt_labels,
                      gt_bboxes_ignore=None)

        buf_data_size = None
        if not self.buffer.is_empty():
            # 这里的返回值顺序可以参考 self.buffer.attributes
            buf_img, buf_img_matas, buf_gt_bboxes, buf_gt_labels = self.buffer.get_data(
                self.minibatch_size, transform=None  # 在目标检测任务中，向buffer中存的就是经过transform的图像，所以此时无需transform
            )
            buf_data_size = buf_img.shape[0]

            buf_img = buf_img.to(img.device)
            buf_gt_labels = [x.to(gt_labels[0].device) for x in buf_gt_labels]
            buf_gt_bboxes = [x.to(gt_bboxes[0].device) for x in buf_gt_bboxes]

            er_loss = super().forward_train(
                buf_img,
                buf_img_matas,
                buf_gt_bboxes,
                buf_gt_labels,
                gt_bboxes_ignore=None
            )
            er_loss = {"er_" + k: v for k, v in er_loss.items()}
            loss.update(**er_loss)

        self.buffer.add_data(
            examples=sup_img,
            img_metas=sup_img_metas,
            gt_bboxes=sup_gt_bboxes,
            gt_labels=sup_gt_labels,
        )

        return loss


@DETECTORS.register_module(force=True)
class ErFreeAnchor(RetinaNet, BaseLearningMethod):
    ALG_NAME = 'er_free_anchor'
    ALG_COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
             backbone,
             neck,
             bbox_head,
             train_cfg=train_cfg,
             test_cfg=test_cfg,
             pretrained=pretrained,
             init_cfg=init_cfg)

        # 因为需要支持mmdet的多卡训练，所以buffer先存在cpu上，需要计算时再放到指定cuda上：
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        if gt_bboxes_ignore is not None:
            raise NotImplementedError

        # 过滤出有标签数据：
        sup_img = None
        sup_img_metas = []
        sup_gt_bboxes = []
        sup_gt_labels = []
        # 过滤出有标签数据：
        for img_idx, img_meta in enumerate(img_metas):
            if img_meta['tag'] == 'sup':
                if sup_img is None:
                    sup_img = img[img_idx].unsqueeze(0)
                else:
                    sup_img = torch.cat([sup_img, img[img_idx].unsqueeze(0)])
                sup_img_metas.append(img_meta)
                sup_gt_bboxes.append(gt_bboxes[img_idx])
                sup_gt_labels.append(gt_labels[img_idx])

        loss = super().forward_train(
              sup_img,
              sup_img_metas,
              sup_gt_bboxes,
              sup_gt_labels,
              gt_bboxes_ignore=None)

        buf_data_size = None
        if not self.buffer.is_empty():
            # 这里的返回值顺序可以参考 self.buffer.attributes
            buf_img, buf_img_matas, buf_gt_bboxes, buf_gt_labels = self.buffer.get_data(
                self.minibatch_size, transform=None  # 在目标检测任务中，向buffer中存的就是经过transform的图像，所以此时无需transform
            )
            buf_data_size = buf_img.shape[0]

            buf_img = buf_img.to(img.device)
            buf_gt_labels = [x.to(gt_labels[0].device) for x in buf_gt_labels]
            buf_gt_bboxes = [x.to(gt_bboxes[0].device) for x in buf_gt_bboxes]

            er_loss = super().forward_train(
                buf_img,
                buf_img_matas,
                buf_gt_bboxes,
                buf_gt_labels,
                gt_bboxes_ignore=None
            )
            er_loss = {"er_" + k: v for k, v in er_loss.items()}
            loss.update(**er_loss)

        self.buffer.add_data(
            examples=sup_img,
            img_metas=sup_img_metas,
            gt_bboxes=sup_gt_bboxes,
            gt_labels=sup_gt_labels,
        )

        return loss


@DETECTORS.register_module(force=True)
class ErSSD(SingleStageDetector, BaseLearningMethod):
    ALG_NAME = 'er_ssd300'
    ALG_COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
                 backbone,
                 neck=neck,
                 bbox_head=bbox_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)

        # 因为需要支持mmdet的多卡训练，所以buffer先存在cpu上，需要计算时再放到指定cuda上：
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        if gt_bboxes_ignore is not None:
            raise NotImplementedError

        # 过滤出有标签数据：
        sup_img = None
        sup_img_metas = []
        sup_gt_bboxes = []
        sup_gt_labels = []
        # 过滤出有标签数据：
        for img_idx, img_meta in enumerate(img_metas):
            if img_meta['tag'] == 'sup':
                if sup_img is None:
                    sup_img = img[img_idx].unsqueeze(0)
                else:
                    sup_img = torch.cat([sup_img, img[img_idx].unsqueeze(0)])
                sup_img_metas.append(img_meta)
                sup_gt_bboxes.append(gt_bboxes[img_idx])
                sup_gt_labels.append(gt_labels[img_idx])

        loss = super().forward_train(
                      sup_img,
                      sup_img_metas,
                      sup_gt_bboxes,
                      sup_gt_labels,
                      gt_bboxes_ignore=None)

        buf_data_size = None
        if not self.buffer.is_empty():
            # 这里的返回值顺序可以参考 self.buffer.attributes
            buf_img, buf_img_matas, buf_gt_bboxes, buf_gt_labels = self.buffer.get_data(
                self.minibatch_size, transform=None  # 在目标检测任务中，向buffer中存的就是经过transform的图像，所以此时无需transform
            )
            buf_data_size = buf_img.shape[0]

            buf_img = buf_img.to(img.device)
            buf_gt_labels = [x.to(gt_labels[0].device) for x in buf_gt_labels]
            buf_gt_bboxes = [x.to(gt_bboxes[0].device) for x in buf_gt_bboxes]

            er_loss = super().forward_train(
                buf_img,
                buf_img_matas,
                buf_gt_bboxes,
                buf_gt_labels,
                gt_bboxes_ignore=None)
            er_loss = {"er_" + k: v for k, v in er_loss.items()}
            loss.update(**er_loss)

        self.buffer.add_data(
            examples=sup_img,
            img_metas=sup_img_metas,
            gt_bboxes=sup_gt_bboxes,
            gt_labels=sup_gt_labels,
        )

        return loss


@DETECTORS.register_module(force=True)
class ErDETR(DETR, BaseLearningMethod):
    ALG_NAME = 'er_detr'
    ALG_COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
                 backbone,
                 bbox_head,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg)

        # 因为需要支持mmdet的多卡训练，所以buffer先存在cpu上，需要计算时再放到指定cuda上：
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        if gt_bboxes_ignore is not None:
            raise NotImplementedError

        # 过滤出有标签数据：
        sup_img = None
        sup_img_metas = []
        sup_gt_bboxes = []
        sup_gt_labels = []
        # 过滤出有标签数据：
        for img_idx, img_meta in enumerate(img_metas):
            if img_meta['tag'] == 'sup':
                if sup_img is None:
                    sup_img = img[img_idx].unsqueeze(0)
                else:
                    sup_img = torch.cat([sup_img, img[img_idx].unsqueeze(0)])
                sup_img_metas.append(img_meta)
                sup_gt_bboxes.append(gt_bboxes[img_idx])
                sup_gt_labels.append(gt_labels[img_idx])

        loss = super().forward_train(
                      sup_img,
                      sup_img_metas,
                      sup_gt_bboxes,
                      sup_gt_labels,
                      gt_bboxes_ignore=None)

        buf_data_size = None
        if not self.buffer.is_empty():
            # 这里的返回值顺序可以参考 self.buffer.attributes
            buf_img, buf_img_matas, buf_gt_bboxes, buf_gt_labels = self.buffer.get_data(
                self.minibatch_size, transform=None  # 在目标检测任务中，向buffer中存的就是经过transform的图像，所以此时无需transform
            )
            buf_data_size = buf_img.shape[0]

            buf_img = buf_img.to(img.device)
            buf_gt_labels = [x.to(gt_labels[0].device) for x in buf_gt_labels]
            buf_gt_bboxes = [x.to(gt_bboxes[0].device) for x in buf_gt_bboxes]

            er_loss = super().forward_train(
                buf_img,
                buf_img_matas,
                buf_gt_bboxes,
                buf_gt_labels,
                gt_bboxes_ignore=None)
            er_loss = {"er_" + k: v for k, v in er_loss.items()}
            loss.update(**er_loss)

        self.buffer.add_data(
            examples=sup_img,
            img_metas=sup_img_metas,
            gt_bboxes=sup_gt_bboxes,
            gt_labels=sup_gt_labels,
        )

        return loss

