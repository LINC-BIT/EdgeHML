import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from sscl.utils.conf import get_device

from ssl_utils.train_utils import get_optimizer


class BaseModel(nn.Module):
    """
    Semi-supervised Continual Learning Model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(BaseModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform

        self.opt = get_optimizer(
            self.net,
            'SGD',
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=0.0005,
        )  # DER方法直接使用普通SGD，因此按照FixMatch论文的特殊SGD设置参数

        # 原mammoth（DER++）中使用的SGD，对DER++效果较好，但加上半监督方法后不如上面较为复杂的SGD：
        # self.opt = SGD(self.net.parameters(), lr=self.args.lr)

        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    # 新增加的方法，在用于训练的无标注样本批上测试准确率：
    def eval_unlabel(self, ul_inputs):
        self.net.eval()
        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.apply_shadow()
        with torch.no_grad():
            inputs, labels = ul_inputs[0], ul_inputs[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct = torch.sum(pred == labels).item()
            total = labels.shape[0]

        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.restore()
        self.net.train()
        return correct / total

class BaseLearningMethod:
    ALG_NAME = None  # 算法模型的名称
    ALG_COMPATIBILITY = []  # 对持续学习场景的兼容性：'class-il', 'domain-il', 'task-il'

    def before_task(self, *args, **kwargs):
        # 在每个 task 训练开始之前的一些工作，
        # 先于 runner.call_hook('before_epoch') 执行
        pass

    def after_task(self, *args, **kwargs):
        # 在每个 task 训练完成之后的一些工作，
        # 在 runner.call_hook('after_epoch') 之后执行

        pass

    def before_task_eval(self, *args, **kwargs):
        # 在每个 task 验证开始之前的一些工作
        pass

    def after_val(self, *args, **kwargs):
        # 在每个 task 验证完成之后的一些工作
        pass

    def extract_unsup_cls_data(self, raw_data: [list, tuple]):
        all_img_metas = []
        all_img_indexes = []
        all_img = None
        for d in raw_data:
            all_img_metas.extend(d['img_metas'])
            all_img_indexes.extend(d['index'])
            if all_img is None:
                all_img = d['img']
            else:
                all_img = torch.cat([all_img, d['img']])
        return {
            'img_metas': all_img_metas,
            'img': all_img,
            'index': all_img_indexes
        }

    def extract_unsup_cls_data_fast(self, raw_data: dict):
        return {
            'img': raw_data['imgs'].view(
                (-1, raw_data['imgs'].shape[2], raw_data['imgs'].shape[3], raw_data['imgs'].shape[4])
            ),
            'index': raw_data['indexes'].view((-1,)),
        }
