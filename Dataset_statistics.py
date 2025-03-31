import copy
import json
import os

import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

label = [line.strip() for line in open("./TransFormer/AckType.txt").readlines()]
t0 = time.time()
train_x = {i: [] for i in range(20)}  # 保存训练集的输入 0:正常序列 1~19：异常序列
train_len = {i: [] for i in range(20)}  # 保存训练集的有效长度 0:正常序列 1~19：异常序列
mean = [6.15414331e+02, 6.71184550e+02, -5.64234181e-01, 5.43246686e-02,
        -1.93837670e-03, -3.82012742e-03, -8.43909917e-02, 3.48834308e-02]
std = [389.03615522, 285.94288187, 8.16800933, 8.06880247, 1.02237252, 1.20312792, 0.69722838, 0.71100907]


def seqs2data(seqs, index, min_len=10, max_len=200):
    if len(seqs) < min_len:
        return
    # seqs:以车辆（发送者）为单位的消息序列集合，每条消息包含10个特征，发送时间、假名、位置xy、速度xy、加速度xy、航向xy
    # 将序列i追加到对应异常序列中
    temp = copy.deepcopy(seqs[:max_len])  # 大于200截断小于200补0
    temp.extend([[0] * len(seqs[0])] * (max_len - len(seqs)))
    seq_len = min(len(seqs), max_len)

    # print("before:\n", temp)
    SumPseudo = len(set([temp[k][1] for k in range(seq_len)]))
    for j in range(seq_len):
        if j != seq_len - 1:
            temp[seq_len - 1 - j][0] -= temp[seq_len - 2 - j][0]
        else:
            temp[seq_len - 1 - j][0] = temp[1][0]
        # if j == 0:
        #     temp[j][0] = seqs[j + 1][0] - seqs[j][0]
        # else:
        #     temp[j][0] -= seqs[j - 1][0]
        temp[j][1] = SumPseudo - 1
        temp[j][2:] = list(map(lambda x: (x[0] - x[1]) / x[2], zip(temp[j][2:], mean, std)))

    train_x[index].append(temp)
    train_len[index].append(seq_len)
    # if index in (0, 9) and temp[-1][2:] != temp[-2][2:] and temp[-2][2:] != temp[-3][2:]:
    #     train_x[0].append(temp)
    # elif index == 12 :  # DelayedMessages
    # elif index != 0:
    #     train_x[index].append(temp)


def createmixall():
    path0 = r'D:\个人文件\车联网安全\VeReMi\VeReMi_Extension_MixAll'
    # 生成所有数据集的pt文件
    for DataType in os.listdir(path0):
        path = path0 + '\\' + DataType  # VeReMi_0_3600_2022-9-11_12_51_1\..
        path1 = path + '\\' + os.listdir(path)[0]  # traceGroundTruthJSON\..
        traffictime = int(DataType[DataType.index("_", 5) + 1:DataType.index("_", 7)]) // 3600
        DatasetType = f'{traffictime}_{traffictime + 1}_MixAll'
        print(f'\n正在生成 {DatasetType} 训练集...')
        CarsLabel = {}  # 异常车辆集
        #  生成异常车辆集
        files = os.listdir(path)[1:]  # path目录下的所有文件
        # print(path)
        for file in files:
            Anormaltype = int(file[file.index("A") + 1:file.index("-", file.index("A"))])
            CarsLabel[int(file[file.index("-", 9) + 1:file.index("-", 10)])] = Anormaltype  # 设置异常车辆标签 ID:Type

        sender_genuine = {i: [] for i in CarsLabel.keys() if CarsLabel[i] == 0}  # 所有正常sender
        sender_anormal = {i: [] for i in CarsLabel.keys() if CarsLabel[i] != 0}  # 所有异常sender
        print(f"正常车辆数：异常车辆数 = {len(sender_genuine)}:{len(sender_anormal)}")
        # message = {}
        for line in open(path1).readlines():
            msg = json.loads(line)
            if msg["sender"] in sender_genuine:
                # 7个特征，标签、发送时间、假名、位置xy、速度xy
                features = [msg["sendTime"], msg["senderPseudo"]] + msg["pos"][:2] + msg["spd"][:2] + msg["acl"][:2] + \
                           msg["hed"][:2]
                sender_genuine[msg["sender"]].append(features)
                # message[msg["messageID"]] = msg["pos"][:2] + msg["spd"][:2]
        # count[0] += len(sender_genuine.items())
        # 生成所有正常序列
        seq_len = 0
        for senderID, seqs in sender_genuine.items():
            # print(f"sender:msg = {senderID}:{len(seqs)}")
            seq_len += len(seqs)
            train_len[0].append(len(seqs))
        print(f"正常车辆平均消息数 = {seq_len / len(sender_genuine.keys())}")

        # 生成异常序列，先遍历所有车辆日志文件，再根据senderID来收集异常消息
        # 需要注意的是，异常车辆发送的都视为异常消息，且不同车辆可能会收集到同一异常消息，因此需要甄别，避免重复

        messageID = set()
        for file in files:
            for line in open(path + "\\" + file).readlines():
                msg = json.loads(line)
                # 满足3个条件才算是异常消息：发送类型为3，sender为异常车辆，messageID没有出现过
                if msg["type"] == 3 and msg["sender"] in sender_anormal and msg["messageID"] not in messageID:
                    sender_anormal[msg["sender"]].append([msg["sendTime"], msg["senderPseudo"]] +
                                                         msg["pos"][:2] + msg["spd"][:2] + msg["acl"][:2] + msg["hed"][
                                                                                                            :2])
                    messageID.add(msg["messageID"])
                    # if message[msg["messageID"]] == [msg["sendTime"], msg["senderPseudo"]] + msg["pos"][:2] + msg["spd"][:2]:
                    #     print("messageID", msg["messageID"])
        for senderID, seqs in sender_anormal.items():
            # print(f"sender:msg = {senderID}:{len(seqs)}")
            seq_len += len(seqs)
            train_len[CarsLabel[senderID]].append(len(seqs))
        print(f"异常车辆平均消息数 = {seq_len / len(sender_anormal.keys())}")

        print(f'{DatasetType} 训练集成完成，用时：{time.time() - t0:.3f}s')
        # print(len(train_y_genuine)-train_y_genuine.count(0), train_y_anormal.count(0))


def createonebyone():
    path0 = r'D:\个人文件\车联网安全\VeReMi\VeReMi_Extension'
    # 生成所有数据集的pt文件
    for AnormalType in os.listdir(path0):
        for Time in os.listdir(path0 + '\\' + AnormalType):
            path = path0 + '\\' + AnormalType + '\\' + Time
            Time = int(Time[Time.index('_') + 1:Time.index('_', 7)]) // 3600
            path1 = path + "\\" + os.listdir(path)[0]

            print(f'\n正在生成 {AnormalType}_0{Time} 训练集...')
            t0 = time.time()
            CarsLabel = {}  # 异常车辆集
            #  生成异常车辆集
            files = os.listdir(path)[1:]  # path目录下的所有文件
            for file in files:
                Anormaltype = int(file[file.index("A") + 1:file.index("-", file.index("A"))]) > 0
                CarsLabel[int(file[file.index("-", 9) + 1:file.index("-", 10)])] = int(
                    Anormaltype)  # 设置异常车辆标签 ID:Type(0/1)

            sender_genuine = {i: [] for i in CarsLabel.keys() if CarsLabel[i] == 0}  # 所有正常sender
            sender_anormal = {i: [] for i in CarsLabel.keys() if CarsLabel[i] != 0}  # 所有异常sender
            print(f"正常车辆数：异常车辆数 = {len(sender_genuine)}:{len(sender_anormal)}")

            # 除了DOs，GT文件均为正常消息，因此GT文件的所有sender都可以生成训练集
            for line in open(path1).readlines():
                msg = json.loads(line)
                if msg["sender"] in sender_genuine:
                    # 10个特征，发送时间、假名、位置xy、速度xy、加速度xy、航向xy
                    features = [msg["sendTime"], msg["senderPseudo"]] + msg["pos"][:2] + \
                               msg["spd"][:2] + msg["acl"][:2] + msg["hed"][:2]
                    sender_genuine[msg["sender"]].append(features)
            # 生成所有正常序列
            seq_len = 0
            for senderID, seqs in sender_genuine.items():
                # print(f"sender:msg = {senderID}:{len(seqs)}")
                seq_len += len(seqs)
                train_len[0].append(len(seqs))
            print(f"正常车辆平均消息数 = {seq_len / len(sender_genuine.keys())}")

            # 生成异常序列，先遍历所有车辆日志文件，再根据senderID来收集异常消息
            # 需要注意的是，异常车辆发送的都视为异常消息，且不同车辆可能会收集到同一异常消息，因此需要甄别，避免重复
            messageID = set()
            for file in files:
                for line in open(path + "\\" + file).readlines():
                    msg = json.loads(line)
                    # 满足3个条件才算是异常消息：发送类型为3，sender为异常车辆，messageID没有出现过
                    if msg["type"] == 3 and msg["sender"] in sender_anormal and msg["messageID"] not in messageID:
                        sender_anormal[msg["sender"]].append(
                            [msg["sendTime"], msg["senderPseudo"]] + msg["pos"][:2] + msg["spd"][:2] + msg["acl"][:2] +
                            msg["hed"][:2])
                        messageID.add(msg["messageID"])
            seq_len = 0

            for senderID, seqs in sender_anormal.items():
                # print(f"sender:msg = {senderID}:{len(seqs)}")
                seq_len += len(seqs)
                train_len[label.index(AnormalType)].append(len(seqs))
            print(f"异常车辆平均消息数 = {seq_len / len(sender_anormal.keys())}")


createmixall()
createonebyone()
#

# train_len = np.array(train_len)
# type_len = {i: [] for i in range(3)}
# for i in range(len(train_len)):
#     train_len[i] = sorted(train_len[i], reverse=False)
#     if "Dos" in label[i] or 'Grid' in label[i]:
#         type_len[1] += train_len[i]
#     elif i!=0:
#         type_len[2] += train_len[i]
#     else:
#         type_len[0] += train_len[i]
#
#     print(f'异常类型:{i} 共有{len(train_len[i])}辆，车辆消息数：{train_len[i][0]}~{train_len[i][-1]}, 平均消息数:{np.mean(train_len[i])}')
#     # 数据集格式：[Cars,[200,10],[200,1],[1],[1]] , 分别表示bsm消息, padding_mask, len, label
#     # torch.save(dataset_i, f".\\Raw_Data\\{i}_{label[i]}_dataset.pt")
#
# type = ['Genuine', 'Dos included', 'Dos not included']
# len_count = {i: [[], []] for i in range(3)}  # 正常、非Dos异常、Dos异常
# for i in range(3):
#     len_set = set(type_len[i])
#     for msg_len in len_set:
#         len_count[i][0] += [msg_len]
#         len_count[i][1] += [type_len[i].count(msg_len)]
#     plt.plot(len_count[i][0], len_count[i][1])
#     plt.title(f"{type[i]}")
#     plt.show()


type_len = {i: [] for i in range(2)}
for i in range(len(train_len)):
    train_len[i] = sorted(train_len[i], reverse=False)
    if "Dos" in label[i] or 'Grid' in label[i]:
        type_len[1] += train_len[i]  # Dos contained
    else:
        type_len[0] += train_len[i]  # Dos not contained

    print(f'异常类型:{i} 共有{len(train_len[i])}辆，车辆消息数：{train_len[i][0]}~{train_len[i][-1]}, 平均消息数:{np.mean(train_len[i])}')
    # 数据集格式：[Cars,[200,10],[200,1],[1],[1]] , 分别表示bsm消息, padding_mask, len, label
    # torch.save(dataset_i, f".\\Raw_Data\\{i}_{label[i]}_dataset.pt")

count_low200=0
for msg_len in range(200):
    count_low200 += type_len[0].count(msg_len)
print(f'Dos contained: 车辆消息数：{min(type_len[1])}~{max(type_len[1])}， 平均消息数:{np.mean(type_len[1])}')
print(f'Dos not contained: 车辆消息数：{min(type_len[0])}~{max(type_len[0])}，小于200：{count_low200/len(type_len[0])}，平均消息数:{np.mean(type_len[0])}')

type = ['Dos_not_contained', 'Dos_contained']
len_count = {i: [[], []] for i in range(2)}  # Dos异常、非Dos异常
for i in range(2):
    len_set = set(type_len[i])
    for msg_len in len_set:
        len_count[i][0] += [msg_len]
        len_count[i][1] += [type_len[i].count(msg_len)]
    plt.plot(len_count[i][0], len_count[i][1])
    # plt.title(f"{type[i]}")
    plt.xlabel('Lengths of message sequences')
    plt.ylabel('Count')
    # plt.show()
    plt.savefig(f'{type[i]}.pdf',dpi=600, bbox_inches = 'tight')