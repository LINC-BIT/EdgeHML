import argparse
import copy
import os
import os.path as osp
import time
import warnings
import setproctitle
from pathlib import Path

import mmcv
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcv import Config, DictAction
from mmcv.utils import get_git_hash, build_from_cfg, collect_env
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
    init_dist
)
from mmcls import __version__
from mmcls.datasets import build_dataloader

from sscl.datasets.utils.base_task import ContinualLearningTasks
from sscl.trainer.task_trainer import TaskTrainer
# from ssl_utils.models.softteacher.ssod.apis import get_root_logger, train_detector
# from ssl_utils.models.softteacher.ssod.datasets import build_dataloader
from ssl_utils.models.softteacher.ssod.utils import patch_config, patch_runner, find_latest_checkpoint, get_root_logger
# from ssl_utils.models.softteacher.ssod.utils.hooks import DistEvalHook

from sscl.utils.conf import set_random_seed
from sscl.necks import *
from sscl.hooks import *
from sscl.backbone.image_classification import *
from sscl.models.image_classification import *
from sscl.datasets.image_classification import *
from sscl.datasets.pipelines import *

def train_detector(
        model,
        dataset: [list, tuple],
        cfg,
        distributed=False,
        validate=True,
        timestamp=None,
        meta=None
):
    logger = get_root_logger(log_level=cfg.log_level)

    # ========== 准备数据，构建 ContinualLearningTasks ===============：
    data_loaders = [
        # Diff from mmdet（这里的dataloader貌似可以继续用）:
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            sampler_cfg=cfg.data.get("sampler", {}).get("train", {}),
            pin_memory=False,
            persistent_workers=True,  # 这个一定要为True
        )
        for ds in dataset
    ]
    seq_det_tasks = ContinualLearningTasks(data_loaders)

    # ========== 构建 mmcv model模型 ===================：
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # ========== 构建 mmcv optimizer ===================：
    optimizer = build_optimizer(model, cfg.optimizer)

    # ========== 构建 mmcv runner ===================：
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
            # Diff from mmdet: batch_processor=None,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config,
            **fp16_cfg,
            distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # ================= 处理验证集数据和 EvalHook ==================：
    val_dataloaders = []
    assert isinstance(cfg.data.val, (list, tuple)), "需要验证集数组（而非单个验证集）以分别验证每个task"
    for val_cfg in cfg.data.val:
        # 注意：下面原来写的是val_cfg.pop("samples_per_gpu", 1)，
        # 会导致验证集的 batch_size 为 1，从而导致在验证时softmax少了batch这一个维度，
        # 因为mmcv会在仅有1张图片时删掉第0个维度，所以不能让batch_size为1：
        val_samples_per_gpu = val_cfg.pop("samples_per_gpu", cfg.data.samples_per_gpu)
        assert val_samples_per_gpu > 1, f"请将验证集的 batch_size 大于 1（cfg.data.val.samples_per_gpu）"
        # if val_samples_per_gpu > 1:
        #     # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        #     val_cfg.pipeline = replace_ImageToTensor(val_cfg.pipeline)
        val_dataset = build_dataset(val_cfg, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            pin_memory=False,
            persistent_workers=True,
        )
        val_dataloaders.append(val_dataloader)

    eval_cfg = cfg.get("evaluation", {})
    eval_cfg["by_epoch"] = False
    eval_hook = build_from_cfg(
        eval_cfg, HOOKS, default_args=dict(dataloader=val_dataloaders)
    )

    runner.register_hook(eval_hook, priority=80)
    seq_det_tasks.set_val_dataloaders(val_dataloaders)

    # =================== 处理用户自定义钩子 =======================：
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # SoftTeacher 作者重写了 runner.save_checkpoint() 方法（应该是因为内部有 teacher
    # 和 student 两个模型，原生 mmdet 不知道存哪个）。
    # 重写是通过 types.MethodType(newMethod, object) 将 newMethod 转为 object 对象
    # 的一个成员方法，然后赋值给 runner.save_checkpoint。
    runner = patch_runner(runner)

    # ====================== checkpoint 进度恢复 ====================：
    resume_from = None
    if cfg.get("auto_resume", False):  # 默认不进行自动恢复
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    trainer = TaskTrainer(runner, seq_det_tasks, task_type="cls")
    train(trainer, cfg)


def train(
    trainer: TaskTrainer,
    cfg
):
    trainer.before_run()
    for task_index in range(len(trainer.tasks)):
        """
        注意，这里的 task_index 和 task_id 的含义并不一样，前者是 task 出现的顺序，
        后者是用户对每个 task 定义的 id，例如 task_id 可以是不连续的 [0,1,5,6,7]
        """
        trainer.run_a_task(task_index, cfg.workflow)
    trainer.after_run()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both "
            "specified, --options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def main():
    # ================= 解析参数 ==========================：

    # setproctitle.setproctitle(
    #     f"第0卡测耗时,感谢!!"
    # )

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    cfg = patch_config(cfg)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # ====================== 初始化环境相关配置 =============================：

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info_dict['MMClassification'] = __version__ + '+' + get_git_hash()[:7]
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info(logger.handlers)
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text
    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # 强制设置 seed
    if args.seed is None:
        args.seed = 0
        args.deterministic = True
    logger.info(f"Set random seed to {args.seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(args.seed, deterministic=args.deterministic)  # 这里没有支持分布式，详见 mmcls/apis/train.py
    cfg.seed = args.seed
    meta["seed"] = args.seed

    meta["exp_name"] = osp.basename(args.config)

    # =================================================================
    # 开始构建模型：
    # Diff from mmdet:
    try:
        model = build_classifier(cfg.model)
    except Exception as e:
        raise Exception(e)
        def get_model(cur_model,
                      pretrained=False,
                      device=None,
                      **kwargs):

            if isinstance(cur_model, Config):
                config = copy.deepcopy(cur_model)
                if pretrained is True and 'load_from' in config:
                    pretrained = config.load_from
            elif isinstance(cur_model, (str, )) and Path(cur_model).suffix:
                config = Config.fromfile(cur_model)
                if pretrained is True and 'load_from' in config:
                    pretrained = config.load_from
            elif isinstance(cur_model, str):
                metainfo = ModelHub.get(cur_model)
                config = metainfo.config
                if pretrained is True and metainfo.weights is not None:
                    pretrained = metainfo.weights
            else:
                raise TypeError('cur_model must be a name, a path or a Config object, '
                                f'but got {type(config)}')

            if pretrained is True:
                warnings.warn('Unable to find pre-defined checkpoint of the cur_model.')
                pretrained = None
            elif pretrained is False:
                pretrained = None

            if kwargs:
                config.merge_from_dict({'model': kwargs})
            config.model.setdefault('data_preprocessor',
                                    config.get('data_preprocessor', None))

            # from mmpretrain.registry import MODELS
            from mmcls.models.builder import MODELS
            cur_model = MODELS.build(config.model)

            dataset_meta = {}
            if pretrained:
                # Mapping the weights to GPU may cause unexpected video memory leak
                # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
                # from mmengine.runner import load_checkpoint
                from mmcv.runner import load_checkpoint
                checkpoint = load_checkpoint(cur_model, pretrained, map_location='cpu')
                if 'dataset_meta' in checkpoint.get('meta', {}):
                    # mmpretrain 1.x
                    dataset_meta = checkpoint['meta']['dataset_meta']
                elif 'CLASSES' in checkpoint.get('meta', {}):
                    # mmcls 0.x
                    dataset_meta = {'classes': checkpoint['meta']['CLASSES']}

            if len(dataset_meta) == 0 and 'test_dataloader' in config:
                # from mmpretrain.registry import DATASETS
                from mmcls.datasets import DATASETS
                dataset_class = DATASETS.get(config.test_dataloader.dataset.type)
                dataset_meta = getattr(dataset_class, 'METAINFO', {})

            cur_model.dataset_meta = dataset_meta
            cur_model.config = config  # save the config in the cur_model for convenience
            cur_model.to(device)
            cur_model.eval()
            return cur_model

        model = get_model("src/sscl/configs/image_classification/Edgehml_cifar100/EdgehmlMAE_Pretrain_HML200_b8_SplitSemiCifar100_task-il_1k.py", pretrained=True)

    model.init_weights()

    # =================================================================
    # 开始构建数据集，注意，由于是持续学习场景，所以 cfg.data.train 有多个数据集：
    # Diff from mmdet:
    datasets = [
        build_dataset(ds) for ds in cfg.data.train
    ]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        # Diff from mmdet:
        new_meta = dict(
            mmcls_version=__version__ + get_git_hash()[:7],
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
        )
        cfg.checkpoint_config.meta = new_meta
        meta.update(new_meta)

    # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES  # 因为每个task中类别可能不一样，所以不能直接取第0个数据集的类别

    model.CLASSES = cfg.data.classes if hasattr(cfg.data, "classes") else datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
