import os.path as osp

import torch.distributed as dist
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, LoggerHook, WandbLoggerHook
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.core import EvalHook as EvalHook_det
from mmdet.apis import multi_gpu_test as multi_gpu_test_det, single_gpu_test as single_gpu_test_det

from mmcls.core import EvalHook as EvalHook_cls
import torch

# ===========================================================
# 图像分类任务的EvalHook：
@HOOKS.register_module(force=True)
class SeqClsEvalHook(EvalHook_cls):
    """
    持续学习任务中，为每个task提供不同的验证集数据
    """
    def __init__(self, dataloader, **kwargs):
        assert isinstance(dataloader, (list, tuple))
        self.dataloaders = dataloader
        super().__init__(dataloader[0], **kwargs)

    def _do_evaluate(self, runner):
        # 根据当前的task_id设置需要eval的验证集：
        assert hasattr(runner, 'task_id')
        self.dataloader = self.dataloaders[runner.task_id]

        """
        注意，原本 IterBasedRunner 是不需要考虑 epoch 的，
        但是 EMAHook 是按 epoch 来调用相应钩子的，所以此处需要
        模拟出 epoch 的周期。
        具体来说，当进入 eval_hook 时，我们认为一个 epoch 结束了，
        并且标记一下 is_next_epoch 状态，告诉 task_trainer 应该切换到
        下一个 epoch 周期了。
        """
        runner.call_hook('after_train_epoch')

        runner.logger.info(f"EvalHook: 即将评估第 [{runner.task_id}] 个 task 的验证集...")
        super(SeqClsEvalHook, self)._do_evaluate(runner)
        setattr(runner, 'is_next_epoch', True)
        torch.cuda.empty_cache()
        return

        """perform evaluation and save ckpt."""
        results = self.test_fn(runner.model, self.dataloader)
        # 有些模型会导致results在cuda上，进而导致evaluate时报错，
        # 所以需要手动转到cpu上：
        results = [x.cpu() for x in results]
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


# ===========================================================
# 普通目标检测算法的EvalHook：
@HOOKS.register_module(force=True)
class SeqDetEvalHook(EvalHook_det):
    """
    持续学习任务中，为每个task提供不同的验证集数据
    """
    def __init__(self, dataloader, *args, **kwargs):
        assert isinstance(dataloader, (list, tuple))
        self.dataloaders = dataloader
        super().__init__(dataloader[0], *args, **kwargs)

    def _do_evaluate(self, runner):
        # 根据当前的task_id设置需要eval的验证集：
        assert hasattr(runner, 'task_id')
        self.dataloader = self.dataloaders[runner.task_id]
        print(f"即将评估第 [{runner.task_id}] 个 task 的验证集...")
        super(SeqDetEvalHook, self)._do_evaluate(runner)

# Soft Teacher有teacher和student两个子模型，所以需要单独实现EvalHook:
@HOOKS.register_module(force=True)
class SeqSubModulesEvalHook(EvalHook_det):
    def __init__(self, dataloader, *args, evaluated_modules=None, **kwargs):
        assert isinstance(dataloader, (list, tuple))
        self.dataloaders = dataloader
        super().__init__(dataloader[0], *args, **kwargs)
        # super().__init__(*args, **kwargs)
        self.evaluated_modules = evaluated_modules

        test_fn = None if 'test_fn' not in kwargs else kwargs['test_fn']
        if test_fn is None:
            # from mmcv.engine import multi_gpu_test
            test_fn = multi_gpu_test_det

        self.broadcast_bn_buffer = False
        self.tmpdir = None
        self.gpu_collect = False


    def before_run(self, runner):
        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        assert hasattr(model, "submodules")
        assert hasattr(model, "inference_on")

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch and self._should_evaluate(runner):
            for hook in runner._hooks:
                if isinstance(hook, WandbLoggerHook):
                    _commit_state = hook.commit
                    hook.commit = False
                if isinstance(hook, LoggerHook):
                    hook.after_train_iter(runner)
                if isinstance(hook, WandbLoggerHook):
                    hook.commit = _commit_state
            runner.log_buffer.clear()

            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.

        # 根据当前的task_id设置需要eval的验证集：
        assert hasattr(runner, 'task_id')
        self.dataloader = self.dataloaders[runner.task_id]
        print(f"当前使用第 {runner.task_id + 1} 个 task 的验证集进行评估")

        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        if is_module_wrapper(runner.model):  # 此时runner.model是MMDataParallel
            model_ref = runner.model.module  # 取出实际的detector（例如ErSoftTeacher）
        else:
            model_ref = runner.model
        if not self.evaluated_modules:
            submodules = model_ref.submodules  # ['teacher', 'student']
        else:
            submodules = self.evaluated_modules  # 可以选择只评估teacher或student
        key_scores = []


        # 对teacher和student都进行评估：
        for submodule in submodules:
            # change inference on
            model_ref.inference_on = submodule
            # results = multi_gpu_test(
            #     runner.model,
            #     self.dataloader,
            #     tmpdir=tmpdir,
            #     gpu_collect=self.gpu_collect,
            # )
            results = single_gpu_test_det(
                runner.model,
                self.dataloader,
                out_dir=tmpdir,
            )
            if runner.rank == 0:
                key_score = self.evaluate(runner, results, prefix=submodule)
                if key_score is not None:
                    key_scores.append(key_score)

        # key_scores中存了teacher和student的结果，将较好的存到checkpoint中：
        if runner.rank == 0:
            runner.log_buffer.ready = True
            if len(key_scores) == 0:
                key_scores = [None]
            best_score = key_scores[0]
            for key_score in key_scores:
                if hasattr(self, "compare_func") and self.compare_func(
                    key_score, best_score
                ):
                    best_score = key_score

            print("\n")
            # runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            if self.save_best:
                self._save_ckpt(runner, best_score)

    def evaluate(self, runner, results, prefix=""):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs
        )
        for name, val in eval_res.items():
            runner.log_buffer.output[(".").join([prefix, name])] = val

        if self.save_best is not None:
            if self.key_indicator == "auto":
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None
