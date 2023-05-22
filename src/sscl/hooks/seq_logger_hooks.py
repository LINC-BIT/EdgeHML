#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

from mmcv.runner import HOOKS, LoggerHook, TensorboardLoggerHook


@HOOKS.register_module()
class SeqTensorboardLoggerHook(TensorboardLoggerHook):
    """
    每个task上回重复迭代1~n次，所以原生的mmcls的hook会重复地写入logger，
    这里将step改为task_id+iter的计数机制
    """
    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            if hasattr(runner, "task_id"):
                current_iter = int(runner.task_id) * int(runner.max_iters) + runner.iter + 1
            else:
                current_iter = runner.iter + 1
        return current_iter
