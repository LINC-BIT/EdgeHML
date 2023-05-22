import torch

from sscl.utils.buffer import Buffer
from sscl.models.utils.base_model import BaseLearningMethod

from mmdet.models import DETECTORS
from ssl_utils.models.softteacher.ssod.models.soft_teacher import SoftTeacher
from ssl_utils.models.softteacher.ssod.utils import log_every_n
from ssl_utils.models.softteacher.ssod.utils.structure_utils import dict_split


@DETECTORS.register_module()
class ErSoftTeacher(SoftTeacher, BaseLearningMethod):
    ALG_NAME = 'er_soft_teacher'
    ALG_COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(ErSoftTeacher, self).__init__(
            model, train_cfg, test_cfg
        )
        # 因为需要支持mmdet的多卡训练，所以buffer先存在cpu上，需要计算时再放到指定cuda上：
        self.buffer = Buffer(train_cfg.buffer_size, torch.device("cpu"))
        self.minibatch_size = train_cfg.minibatch_size

    def forward_train(self, img, img_metas, **kwargs):
        loss = super().forward_train(img, img_metas, **kwargs)

        buf_data_size = None
        if not self.buffer.is_empty():
            buf_img, buf_img_metas, buf_gt_bboxes, buf_gt_labels = self.buffer.get_data(
                self.minibatch_size, transform=None  # 在目标检测任务中，向buffer中存的就是经过transform的图像，所以此时无需transform
            )
            buf_data_size = buf_img.shape[0]
            buf_img = buf_img.to(img.device)
            buf_gt_labels = [x.to(img.device) for x in buf_gt_labels]
            buf_gt_bboxes = [x.to(img.device) for x in buf_gt_bboxes]

            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in buf_gt_bboxes]) / len(buf_gt_bboxes)}
            )
            er_loss = self.student.forward_train(
                img=buf_img,
                img_metas=buf_img_metas,
                gt_bboxes=buf_gt_bboxes,
                gt_labels=buf_gt_labels,
            )
            er_loss = {"er_" + k: v for k, v in er_loss.items()}
            loss.update(**er_loss)

        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")
        if "sup" in data_groups:
            self.buffer.add_data(
                examples=data_groups["sup"]["img"],
                img_metas=data_groups["sup"]["img_metas"],
                gt_bboxes=data_groups["sup"]["gt_bboxes"],
                gt_labels=data_groups["sup"]["gt_labels"],
            )

        return loss

    def after_task(self):
        # 当前task结束，在进行评估之前，切换为student模型：
        self.inference_on = "student"
