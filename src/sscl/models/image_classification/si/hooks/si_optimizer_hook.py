import os.path as osp

import torch.distributed as dist
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, LoggerHook, WandbLoggerHook
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.runner.hooks import OptimizerHook
from torch.nn.utils import clip_grad


@HOOKS.register_module(force=True)
class SiOptimizerHook(OptimizerHook):

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_value_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())

        runner.optimizer.step()

        net = runner.model if not is_module_wrapper(runner.model) \
            else runner.model.module  # 这里是为了支持某些算法需要的“子模型”
        net.small_omega += runner.optimizer.defaults['lr'] * net.get_grads().data ** 2

