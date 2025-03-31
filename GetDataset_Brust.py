import copy
import numpy as np
import pandas as pd
import time
from collections import Counter
import torch
from torch.utils.data import TensorDataset

t0 = time.time()
path = r'G:\Chrome\BurST-ADMA_v0.1.csv'
df = pd.read_csv(path)
n_labels = len(set(df['label']))
train_x = {i: [] for i in range(n_labels)}  # 保存训练集的输入 0:正常序列 1~7：异常序列
train_len = {i: [] for i in range(n_labels)}  # 保存训练集的有效长度 0:正常序列 1~7：异常序列


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
    path = r'G:\Chrome\BurST-ADMA_v0.1.csv'
    df = pd.read_csv(path)
    id = {elem for elem in df['id'] if 'ped' not in elem}
    labels = set(df['label'])
    element_count = Counter(df['label'].to_list())
    print(f'各label消息统计:{element_count}')
    seqs_id = {i: [] for i in id}
    id2label = {i: [] for i in id}
    seqs_label = {i: [] for i in labels}
    # 提取第2到第7列的数据
    data = df.values.tolist()
    subset_df = df.iloc[:, 2:7]
    subset_df['speed'], subset_df['heading'] = subset_df['heading'], subset_df['speed']
    mean = subset_df.mean().tolist()
    std = subset_df.var().tolist()
    # print(mean, std)
    # timestep id x y heading speed acceleration label
    for row in data:
        if 'ped' not in row[1]:
            seqs_id[row[1]].append([row[0]] + row[2:4] + [row[5]] + [row[4]] + [row[6]])
            id2label[row[1]] = row[-1]

    for id, seq in seqs_id.items():
        seqs_label[id2label[id]].append(seq)

    key_lengths = {k: len(v) for k, v in seqs_label.items()}
    print(f'总车辆数：{len(seqs_id.values())}，各label车辆统计:{key_lengths}')

    length = {i: [] for i in labels}
    for k, v in seqs_label.items():
        for s in v:
            length[k].append(len(s))
        print(f'label{k}平均长度:{sum(length[k]) / len(length[k])}')

    for label, seqs in seqs_label.items():
        for seq in seqs:
            index = 0
            while index < len(seq):
                # print(len(seq[0]))
                seqs2data(seq[index:index + max_len], label, mean, std)
                index += max_len
def map_numbers(num):
    if num == 0:
        return 0
    elif num >= 1 and num <= 8 or num == 12:
        return 1
    elif num >= 9 and num <= 11 or num >= 13 and num <= 19:
        return 2
    else:
        return None
max_len = 200
create(max_len)

# train_x[i]:[Cars,200,100]
key_padding_mask = {i: [] for i in range(n_labels)}
for i in range(len(train_len)):
    for j in range(len(train_len[i])):
        x = ([False] * train_len[i][j])
        x.extend([True] * (max_len - train_len[i][j]))
        key_padding_mask[i].append(x)

label = [line.strip() for line in open("Dataset/LuST/Splited_Dataset/AckType.txt").readlines()]
# key_padding_mask = [([[1] * msg_size] * train_len[i]).extend([[0] * msg_size] * (max_len - train_len[i])) for i in range(len(train_len))]
for i in range(len(train_x)):
    dataset_i = TensorDataset(torch.tensor(train_x[i]), torch.tensor(key_padding_mask[i]),
                              torch.tensor(train_len[i]), torch.tensor(len(train_x[i]) * [[int(i > 0), map_numbers(i), i]]))
    print(f'异常类型:{i} 共有{len(dataset_i)}条')
    # dataset_i: [Cars,[200,10],[200,1],[1],[1]], bsm, padding_mask, len, label respectively
    torch.save(dataset_i, f".\\Dataset\\BurST\\Raw_Data\\{i}_{label[i]}_dataset.pt")
