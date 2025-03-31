import os
import sys
from typing import Tuple, Any

from HMCFormer.base.torchseq_dataset import TorchSeqDataset
import torch
import numpy as np
import torch.utils.data as data



class VeReMi_V2_Dataset(TorchSeqDataset):

    def __init__(self, root: str = r'../VeReMi_splited_dataset/', msg_size: int = 10):
        super().__init__(root)
        # Define normal and outlier classes
        # Get train set
        self.train_set = VeReMi_V2(root=self.root, msg_size=msg_size, type='train')
        # Get test set
        self.test_set = VeReMi_V2(root=self.root, msg_size=msg_size, type='eval')
        self.n_classes = self.train_set.n_classes



class VeReMi_V2(data.Dataset):
    def __init__(self, root: str = None, msg_size: int = 10, type='train', anor_size: int = 250):
        super().__init__()
        # Define the properties of the dataset
        self.root = root
        self.data = []
        self.padding_mask = []
        self.seq_len = []  # 有效长度
        self.targets = []  # 数据集的原始标签

        # load train_dataset/eval_dataset 建议预先划分好train_dataset/eval_dataset，并做标准化处理
        dataset = []
        if type == 'train':
            root += type
            for file in os.listdir(root):
                dataset += torch.load(root + '/' + file)
        elif type == 'eval':
            root += type
            for file in os.listdir(root):
                dataset += torch.load(root + '/' + file)
        elif type == 'test':
            root += type
            for file in os.listdir(root):
                dataset += torch.load(root + '/' + file)

        self.type_size = {i: 0 for i in range(20)}
        # dataset = getdataset(genuine_train_size=50000, anor_size=100, train=train)
        # self.seq_len.append(dataset[:, 2].int())
        for seq in dataset:
            self.data.append(seq[0].numpy()[:, :msg_size])  # [:, :6]
            self.padding_mask.append(seq[1].numpy())
            self.seq_len.append(seq[2].int())
            self.targets.append(seq[3].numpy())
            self.type_size[int(seq[3][-1])] += 1
            # print("Sample size: ", seq[0].size(), seq[1].size(), seq[2], seq[3])


        self.data = torch.as_tensor(np.array(self.data))
        self.padding_mask = torch.as_tensor(np.array(self.padding_mask))
        # print(self.data.size())
        self.seq_len = torch.as_tensor(self.seq_len, dtype=torch.int64)
        self.max_len = self.data.size()[1]
        self.fea_size = self.data.size()[2]
        unique_elements, _ = np.unique(self.targets, return_counts=True)
        self.n_classes = len(unique_elements)
        self.targets = torch.as_tensor(self.targets, dtype=torch.int64)


    def __getitem__(self, index) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sequence, target, semi_target, index)
        """
        seq, padding_mask, seq_len, target = \
            self.data[index], self.padding_mask[index], self.seq_len[index], self.targets[index]

        return seq, padding_mask, seq_len, target, index

    def __len__(self) -> int:
        return len(self.targets)
