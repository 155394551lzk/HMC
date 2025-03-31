import copy
import json
import os

import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import TensorDataset

Targets = [0, 1, 2, 3, 4, 5]
t0 = time.time()
train_x = {i: [] for i in Targets}  # 保存训练集的输入 0:正常序列 1~5：异常序列
train_len = {i: [] for i in Targets}  # 保存训练集的有效长度 0:正常序列 1~5：异常序列

import scipy.io

mean = [5098.412167792616, 5947.270528614088, 0.03795698304290086, 1.3764345281842867]
std = [3303.883147415046, 2343.195544146264, 8.243285890981868, 19.729601843225012]


def seqs2data(seqs, label, mean, std, min_len=10, max_len=200):
    if len(seqs) < min_len:
        return
    temp = copy.deepcopy(seqs[:max_len])  # 大于200截断
    seq_len = min(len(seqs), max_len)

    for j in range(seq_len):
        # 将发送时间处理为发送间隔
        if j != seq_len - 1:
            temp[seq_len - 1 - j][0] -= temp[seq_len - 2 - j][0]
        else:
            temp[seq_len - 1 - j][0] = temp[1][0]
        temp[j][1:] = list(map(lambda x: (x[0] - x[1]) / x[2], zip(temp[j][1:], mean, std)))  # Z-score标准化
    temp.extend([[0] * len(seqs[0])] * (max_len - len(seqs)))  # 小于200补0
    train_x[label].append(temp)  # 将消息序列追加至训练集中
    train_len[label].append(seq_len)  # 将消息长度追加至训练集中


def create(max_len=200):
    root = r'H:\personal\IoV_Security\VeReMi\VeReMi\WiSec_DataModifiedVeremi_Dataset-master'
    print(os.getcwd())
    # root = r'/home/VeReMi/WiSec_DataModifiedVeremi_Dataset-master'
    for file in os.listdir(root):
        if 'attack' in file:
            path = root + '/' + file  # VeReMi_0_3600_2022-9-11_12_51_1\..
            mat = scipy.io.loadmat(path)[f'{file[:-4]}']
            
            # timestep id pox_x pos_y speed_x speed_y label
            data = mat[:, [1, 7, 9, 10, 12, 13, 16]].astype(np.float32)
            # data = np.array(data)
            data[np.isnan(data)] = 0
            data[:, 1] = data[:, 1].astype(int)
            # 创建一个条件数组来标记需要忽略的位置
            mask = (data[:, 6] != 0)
            data[:, 6] = np.where(mask, np.log2(data[:, 6]) + 1, 0)
            # print(data[:, 6])
            labels = set(data[:, -1])
            # print(labels)
            ids = set(data[:, 1])

            element_count = Counter(data[:, -1].tolist())
            print(f'各label消息统计:{element_count}')
            seqs_id = {i: [] for i in ids}
            id2label = {i: [] for i in ids}
            seqs_label = {i: [] for i in labels}

            # mean = np.mean(data, axis=0).tolist()
            # std = np.std(data, axis=0).tolist()
            # print(mean, std)
            data = data.tolist()
            for row in data:
                seqs_id[row[1]].append([row[0]] + row[2:6])
                if id2label[row[1]] is not None and id2label[row[1]] and id2label[row[1]] != row[6]:
                    print('id2label', id2label[row[1]])
                id2label[row[1]] = row[6]

            # for k,v in id2label.items():
            #     if len(set(v))>1:
            #         print(v)

            for id, seq in seqs_id.items():
                seqs_label[id2label[id]].append(seq)

            key_lengths = {k: len(v) for k, v in seqs_label.items()}
            print(f'总车辆数：{len(seqs_id.values())}，各label车辆:{key_lengths}')

            length = {i: [] for i in labels}
            for k, v in seqs_label.items():
                for s in v:
                    length[k].append(len(s))
                # print(length[k])
                print(f'label{k} 平均长度:{sum(length[k]) / len(length[k])}')

            s, l = 0, 0
            for k, v in length.items():
                s+=sum(length[k])
                l+=len(length[k])
            print(f'ave. length:{s/l}')
            for label, seqs in seqs_label.items():
                for seq in seqs:
                    index = 0
                    while index < len(seq):
                        # print(len(seq[0]))
                        seqs2data(seq[index:index + max_len], label, mean, std)
                        index += 50


max_len = 200
create(max_len)

# train_x[i]:[Cars,200,100]
n_labels = 6
key_padding_mask = {i: [] for i in Targets}
for i in Targets:
    for j in range(len(train_len[i])):
        x = ([False] * train_len[i][j])
        x.extend([True] * (max_len - train_len[i][j]))
        key_padding_mask[i].append(x)

label = [line.strip() for line in open("Dataset/VeReMi/Splited_Dataset/AckType.txt").readlines()]
j = 0
# key_padding_mask = [([[1] * msg_size] * train_len[i]).extend([[0] * msg_size] * (max_len - train_len[i])) for i in range(len(train_len))]
for i in Targets:
    dataset_i = TensorDataset(torch.tensor(train_x[i]), torch.tensor(key_padding_mask[i]),
                              torch.tensor(train_len[i]),
                              torch.tensor(len(train_x[i]) * [[int(i > 0), i]]))
    print(f'异常类型:{i} 共有{len(dataset_i)}条')
    # dataset_i: [Cars,[200,10],[200,1],[1],[1]], bsm, padding_mask, len, label respectively
    torch.save(dataset_i, f"./Dataset/VeReMi/Raw_Data/{i}_{label[j]}_dataset.pt")
    j += 1
