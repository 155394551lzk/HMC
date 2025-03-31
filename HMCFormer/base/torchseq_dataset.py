from .base_dataset import BaseDataset
from torch.utils.data import DataLoader


class TorchSeqDataset(BaseDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 4) -> (
            DataLoader, DataLoader):
        # print("num_workers: ", num_workers)
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=False)  #数据集若不能被batch_size整除则丢弃余数
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
