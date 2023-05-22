#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""

from mmcv.runner.hooks import OptimizerHook, HOOKS

@HOOKS.register_module(force=True)
class TwoStageSemiSupOptimizerHook(OptimizerHook):
    """
    先用无监督样本loss更新backbone，然后用有监督样本loss更新head
    """

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        # 第一阶段：
        if 'loss_head' in runner.outputs['losses']:
            # runner.model.module.backbone.freeze_backbone()

            if self.detect_anomalous_params:
                self.detect_anomalous_parameters(runner.outputs['losses']['loss_head'], runner)
            runner.outputs['losses']['loss_head'].backward(retain_graph=True)

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])

            runner.model.module.backbone.clear_grad(num_stage=3)
            # runner.model.module.backbone.unfreeze_backbone()

        # 第二阶段：
        if 'loss_all' in runner.outputs['losses']:
            if self.detect_anomalous_params:
                self.detect_anomalous_parameters(runner.outputs['losses']['loss_all'], runner)
            runner.outputs['losses']['loss_all'].backward()

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])

        runner.optimizer.step()
