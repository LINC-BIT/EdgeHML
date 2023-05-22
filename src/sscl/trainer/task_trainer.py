#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
"""
import numpy as np
import warnings
import time
import mmcv
from mmcv.runner.utils import get_host_info
from mmcv.runner.iter_based_runner import IterLoader
from mmcv.parallel import is_module_wrapper
from mmdet.apis import single_gpu_test as single_gpu_test_det
from mmcls.apis import single_gpu_test as single_gpu_test_cls
from copy import deepcopy


class IterLoaderFast(IterLoader):

    def __init__(self, dataloader, fast_mode: bool = True):
        super(IterLoaderFast, self).__init__(dataloader)

        # 当迭代次数很高的时候，为了避免可能的死锁问题，可以将 fast_mode 设为 False：
        self.fast_mode = fast_mode
        self.sleep_time = 0.1 if fast_mode else 2

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            # 原本是2s，改为0.1s
            time.sleep(self.sleep_time)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data


class TaskTrainer:
    """
    封装了 mmcv 中的 runner，
    理论上应该兼容 mmcv 下的所有库（除了 _simple_test 方法）
    """

    def __init__(self, runner, tasks, task_type="det"):
        """

        :param runner: mmdet 中的 runner
        :param tasks: 可以在此直接传入所有 task 的 data_loader，
            tasks 只要实现 __getitem__ 以能够获取第 index 个 task 的数据即可。
        :param type: "classification" or "detection"
        """
        self.runner = runner
        setattr(self.runner, 'task_id', -1)
        self.previous_ids = []  # 记录已经训练过的 task，因为每个 task 训练后都需要在之前的 task 上测试
        self.tasks = tasks
        sorted(self.tasks.train_data_loaders, key=(lambda x: x.dataset.task_id))
        sorted(self.tasks.val_data_loaders, key=(lambda x: x.dataset.task_id))

        self.result_matrix = []
        self.result_matrix_class_il = []

        if task_type[:3] == "cla" or task_type[:3] == "cls":
            self.task_type = "cls"
            self.metric_name = "Acc"
        elif task_type[:3] == "det":
            self.task_type = "det"
            self.metric_name = "mAP"
        else:
            raise NotImplementedError

    def before_run(self):
        """
        在所有 task 执行之前执行这里，
        如果子类需要重写 before_run 的逻辑，需要先 super().before_run()
        :return:
        """

        work_dir = self.runner.work_dir if self.runner.work_dir is not None else 'NONE'
        self.runner.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.runner.logger.info('Hooks will be executed in the following order:\n%s',
                         self.runner.get_hook_info())

        self.runner.call_hook('before_run')

        # ================== 在所有task上先测试一遍 ==============：
        net = self.runner.model if not is_module_wrapper(self.runner.model) \
            else self.runner.model.module  # 这里是为了支持某些算法需要的“子模型”

        if hasattr(net, "SKIP_TEST_BEFORE_START") and net.SKIP_TEST_BEFORE_START is True:
            self.runner.logger.info(f"当前模型 {net.ALG_NAME} 指定 SKIP_TEST_BEFORE_START == True，跳过测试")
        else:
            result_before_start = []
            for _, val_dataloader in self.tasks:
                if hasattr(net, "before_task_eval"):
                    net.before_task_eval(runner=self.runner)

                task_ap, metric, _ = self._simple_test(val_dataloader)
                result_before_start.append(task_ap)
                # self.runner.logger.info(metric)

                if hasattr(net, "after_val"):
                    net.after_val(runner=self.runner)

            res = sum(result_before_start) / len(result_before_start)
            self.runner.logger.info(f"Test on all tasks, {self.metric_name} = {round(res, 3)}")

    def _compute_statistic(self, acc_matrix):
        max_history_acc = deepcopy(acc_matrix[-1])
        for cur_task_results in acc_matrix:
            for j, acc_when_cur_task in enumerate(cur_task_results):
                if acc_when_cur_task > max_history_acc[j]:
                    max_history_acc[j] = acc_when_cur_task
        forget_vals = [max_acc - final for max_acc, final in zip(max_history_acc, acc_matrix[-1])]
        avg_forget_val = sum(forget_vals) / len(forget_vals)

        avg_accs = [sum(vals)/len(vals) for vals in acc_matrix]
        mean_avg_acc = sum(avg_accs) / len(avg_accs)

        self.runner.logger.info(
            f"Average Forget: {round(avg_forget_val,2)}, Mean Average {self.metric_name}: "
            f"{round(mean_avg_acc,2)} {[round(x,2) for x in avg_accs]}")

    def after_run(self):
        """
        在所有 task 执行完成之后执行这里，
        如果子类需要重写 after_run 的逻辑，需要先 super().after_run()
        :return:
        """
        self.runner.call_hook('after_run')

        # 根据result_matrix统计指标：
        self.runner.logger.info(f"=========================================")
        self._compute_statistic(self.result_matrix)
        if len(self.result_matrix_class_il[0]) > 0:
            self.runner.logger.info(f"=========================================")
            self.runner.logger.info(f"class-il:")
            self._compute_statistic(self.result_matrix_class_il)
        self.runner.logger.info(f"=========================================")

    def run_a_task(self, task_index, workflow=None, max_iters=None,  **kwargs):
        """
        封装了 mmdet 中 runner.run() 方法，
        该方法通常不需要用户重写（除非需要实现特殊的逻辑）

        :param task_id: 当前task的index（index不等同于task_id，但类似）
        :param workflow:
        :param max_iters:
        :param kwargs:
        :return:
        """

        # =============== 处理 workflow ======================：
        if workflow is None:
            workflow = [("train", 1)]
        assert mmcv.is_list_of(workflow, tuple)

        # =============== 处理 max_iters =====================：
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self.runner._max_iters = max_iters
        assert self.runner._max_iters is not None, (
            'max_iters must be specified during instantiation')

        self.runner.logger.info('workflow: %s, max: %d iters', workflow,
                         self.runner._max_iters)

        # ================ 处理 data_loader =====================：
        assert task_index < len(self.tasks)
        data_loader = self.tasks[task_index][0]
        task_id = data_loader.dataset.task_id
        setattr(self.runner, "task_id", task_id)  # 设置runner的状态，用于EvalHook选择相应的验证集
        if task_id in self.previous_ids:
            self.runner.logger.warning(f"Current task [{task_id}] has appeared before")
        else:
            self.previous_ids.append(task_id)

        if not isinstance(data_loader, list):
            data_loader = [data_loader]

        net = self.runner.model if not is_module_wrapper(self.runner.model) \
            else self.runner.model.module  # 这里是为了支持某些算法需要的“子模型”

        # ============== 在 task 开始之前执行一些逻辑 ================：

        if hasattr(net, "before_task"):
            net.before_task(dataloaders=data_loader, num_iters=self.runner._max_iters, runner=self.runner)

        self.runner.call_hook('before_epoch')

        # runner.iter在每个task开始之前被重置
        self.runner._iter = 0

        # ============== 构造 dataloader ================：
        # 注意，需要在完成 before_task 之后再构造数据集，
        # 因为 before_task 可能对 data_loader 有改动
        assert len(data_loader) == len(workflow)
        iter_fast_mode = net.iter_fast_mode if hasattr(net, "iter_fast_mode") else True
        iter_loaders = [IterLoaderFast(x, iter_fast_mode) for x in data_loader]

        # ================= 开始训练 ===========================：
        self._run_iters(workflow, iter_loaders, **kwargs)

        # ============= 在 task 上训练完之后，处理一些事项 ===============：
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.runner.call_hook('after_epoch')

        if hasattr(net, "after_task"):
            net.after_task(dataloaders=data_loader, runner=self.runner)

        # ======== 对当前task_id及之前的task进行评估，计算平均准确率 ======：
        result_so_far = []
        result_so_far_class_il = []
        for tid in self.previous_ids:
            val_dl = self.tasks[tid][1]

            setattr(self.runner, "task_id", tid)
            if hasattr(net, "before_task_eval"):
                net.before_task_eval(runner=self.runner)

            task_ap, metric, res_class_il = self._simple_test(val_dl)
            # self.runner.logger.info(metric)
            result_so_far.append(task_ap)
            if res_class_il is not None:
                result_so_far_class_il.append(res_class_il[0])

            if hasattr(net, "after_val"):
                net.after_val(runner=self.runner)

        self.runner.logger.info("========================================================")
        self.runner.logger.info(
            f"Test on task {self.previous_ids}, {self.metric_name} = "
            f"{round(sum(result_so_far)/len(result_so_far), 2)}, "
            f"{[round(x, 2) for x in result_so_far]}")
        if len(result_so_far_class_il) > 0:
            self.runner.logger.info("========================================================")
            self.runner.logger.info(
                f"Test on task {self.previous_ids} (class-il), {self.metric_name} = "
                f"{round(sum(result_so_far_class_il)/len(result_so_far_class_il), 2)}, "
                f"{[round(x, 2) for x in result_so_far_class_il]}")
        self.runner.logger.info("========================================================")

        self.result_matrix.append(result_so_far)
        self.result_matrix_class_il.append(result_so_far_class_il)

        return result_so_far

    def _simple_test(self, val_dataloader):
        """
        在传入的验证集上进行测试

        :param val_dataloader:
        :return: cur_ap: 当前 task 的所有类别上的平均 AP
                 metric: OrderedDict([
                            ('bbox_mAP', 0.0),
                            ('bbox_mAP_50', 0.0),
                            ('bbox_mAP_75', 0.0),
                            ('bbox_mAP_s', 0.0),
                            ('bbox_mAP_m', 0.0),
                            ('bbox_mAP_l', 0.0),
                            ('bbox_mAP_copypaste', '0.000 0.000 0.000 0.000 0.000 0.000')
                        ])
        """

        res_on_cur_task = 0.0

        if self.task_type == "cls":
            outputs = single_gpu_test_cls(
                self.runner.model,
                val_dataloader,
                show=False,
                out_dir=None
            )

            # 有些模型会导致results在cuda上，进而导致evaluate时报错，
            # 所以需要手动转到cpu上。但是现在修复了模型自己的bug，不需要这句了
            # outputs = [x.cpu() for x in outputs]

            # outputs就是 2000张图片*10个类别上的概率；
            # 根据持续学习的场景，对 outputs 进行处理：
            assert hasattr(val_dataloader.dataset, "CL_TYPE"), "请在数据集中指定其对应的持续学习场景 `CL_TYPE`"
            cl_type = val_dataloader.dataset.CL_TYPE
            # ================== 检查持续学习场景兼容性 ==============：
            assert cl_type in self.runner.model.module.ALG_COMPATIBILITY, \
                f"当前 model ({self.runner.model.module.ALG_COMPATIBILITY}) 不支持该任务数据集 ({cl_type})"

            print(f"当前 task 测试模式：{cl_type}")
            if str(cl_type).lower() == "task-il":
                retained_classes = val_dataloader.dataset.class_index
                masked_outputs = []
                for op in outputs:
                    masked_op = np.array(op)
                    masked_op[[c for c in range(len(val_dataloader.dataset.CLASSES)) if c not in retained_classes]] = -float('inf')
                    masked_outputs.append(
                        masked_op
                    )
                # outputs = masked_outputs

            elif str(cl_type).lower() == "class-il":
                masked_outputs = outputs

            elif str(cl_type).lower() == "domain-il":
                masked_outputs = outputs
            else:
                raise NotImplementedError

            metric = val_dataloader.dataset.evaluate(
                masked_outputs, logger=self.runner.logger
            )
            res_on_cur_task = metric['accuracy_top-1']

            if str(cl_type).lower() == "task-il":
                # 如果是task-il的话，也顺便测一下class-il：

                metric_class_il = val_dataloader.dataset.evaluate(
                    outputs, logger=self.runner.logger
                )
                res_on_cur_task_class_il = metric_class_il['accuracy_top-1']

                return res_on_cur_task, metric, (res_on_cur_task_class_il, metric_class_il)

        elif self.task_type == "det":
            outputs = single_gpu_test_det(
                self.runner.model,
                val_dataloader,
                show=False,
                out_dir=None
            )

            # 这里得到的outputs是n张图片的list，每张图片是80个类别的list，
            # 每个类别是x*5个bbox结果。
            # 在 task-il 场景下，需要屏蔽掉一些类别

            metric = val_dataloader.dataset.evaluate(
                outputs, logger=self.runner.logger, classwise=True
            )

            # 每个类别的 AP：
            ap_on_cur_task = 0.0
            if hasattr(val_dataloader.dataset, "results_per_category"):
                results_per_category = val_dataloader.dataset.results_per_category
                cur_aps = []
                cur_cls = []
                for cls_name, cls_ap in results_per_category:
                    if cls_ap != 'nan':  # 排除掉非当前验证集的类别
                        cur_aps.append(float(cls_ap))
                        cur_cls.append(cls_name)
                self.runner.logger.info(f"当前 task 上的类别：{cur_cls}")
                if len(cur_aps) == 0:
                    self.runner.logger.warning(f"Current val_dataloder has no AP result")
                    ap_on_cur_task = 0
                else:
                    ap_on_cur_task = sum(cur_aps) / len(cur_aps)
                delattr(val_dataloader.dataset, "results_per_category")
            else:
                self.runner.logger.warning(f"No classwise results")

            res_on_cur_task = ap_on_cur_task

        else:
            raise NotImplementedError

        return res_on_cur_task, metric, None

    def _run_iters(self, workflow, iter_loaders, **kwargs):
        # ======== 开始在当前 task 上训练 ========：
        while self.runner.iter < self.runner._max_iters:
            if self.runner.iter == 0 or (hasattr(self.runner, 'is_next_epoch') and self.runner.is_next_epoch is True):
                self.runner.logger.info(f"进入了新的 virtual epoch")
                self.runner.call_hook('before_train_epoch')
                setattr(self.runner, 'is_next_epoch', False)

            for i, flow in enumerate(workflow):
                self.runner._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self.runner, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                            format(mode))
                iter_runner = getattr(self.runner, mode)
                for _ in range(iters):
                    if mode == 'train' and self.runner.iter >= self.runner._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        if self.runner.is_next_epoch is False:
            # 加一次判断，防止没有调用eval_hook（也就没有调用after_train_epoch）就直接开始下一阶段了
            setattr(self.runner, 'is_next_epoch', True)
            self.runner.call_hook('after_train_epoch')


