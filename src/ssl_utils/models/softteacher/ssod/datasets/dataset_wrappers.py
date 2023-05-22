from mmdet.datasets import DATASETS, ConcatDataset, build_dataset


@DATASETS.register_module(force=True)
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        # build 两个 dataset，分别代表有标签数据集和无标签数据集
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    # 因为在构造函数中将有标签和无标签数据拼接为一个dataset，
    # 所以后续要区分不同分支，故还额外新增了两个属性sup、unsup：
    @property  # 设置属性，方便处理
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]
