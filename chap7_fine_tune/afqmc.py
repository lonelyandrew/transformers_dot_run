import jsonlines
from torch.utils.data import Dataset


class AFQMC(Dataset):
    """AFQMC数据集.

    AFQMC (Ant Financial Question Matching Corpus) ：蚂蚁金融语义相似度数据集，该数据集由蚂蚁金服提供。
    """

    def __init__(self, data_file: str) -> None:
        """初始化数据集.

        Args:
            data_file: 数据集文件路径.
        """
        self.data = self.load_data(data_file)

    def load_data(self, data_file: str) -> dict[int, dict[str, str]]:
        """加载数据集.

        Args:
            data_file: 数据集文件路径.

        Returns:
            返回一个样本字典, key为索引, value为样本数据.
        """
        data: dict[int, dict[str, str]] = {}

        with jsonlines.open(data_file) as reader:
            for idx, sample in enumerate(reader):
                data[idx] = dict(sample)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]