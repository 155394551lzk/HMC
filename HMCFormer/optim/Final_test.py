import json
import math
import os
import random

from torch import nn
from torch.backends import cudnn

import pandas as pd
from torch.utils.data import DataLoader

import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import time
import torch
import numpy as np
from HMCFormer.Parameters import Parameters
from HMCFormer.datasets.VeReMi_V2 import VeReMi_V2
from HMCFormer.networks.contrast import ContrastModel
from HMCFormer.networks.main import build_network
from torch.utils.data import Subset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True  # 保证CNN的可复现性
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


class Final_test():
    def __init__(self, args: Parameters = None):
        super(Final_test, self).__init__()
        # Optimization parameters
        self.net = None
        # Results
        self.auc = 0.
        self.test_time = 0.
        self.AttackIDs = []
        self.res = []
        self.acc = []
        self.rec = []
        self.pre = []
        self.f1 = []


    def evaluate(self, pre_y, true_y, datasetpath, average='macro'):
        # 计算F1,rec,prec,acc
        classes = [line.strip() for line in open(f"{datasetpath}/AckType.txt").readlines()]
        f1 = []
        rec = []
        prec = []
        acc = []
        features = sorted(set(true_y))
        for label in features:
            tp = np.sum((true_y == label) & (pre_y == label))
            tn = np.sum((true_y != label) & (pre_y != label))
            fp = np.sum((true_y != label) & (pre_y == label))
            fn = np.sum((true_y == label) & (pre_y != label))
            # p = tp/(tp+fp) r = tp/(tp+fn)
            f1.append(100*2 * tp / (2 * tp + fp + fn))
            rec.append(100*tp / (tp + fn))
            prec.append(100*tp / (tp + fp))
            acc.append((100 * (tp + tn)) / (tp + tn + fn + fp))
            # print(f"\t 预测标签{i}：{(pre_y_all == i).sum()} 预测正确:预测错误={tp}:{fp}")
            print(
                f'\t{label}:{classes[label]} | {sum(true_y==label)} | Acc: {acc[-1]:.3f} | F1: {f1[-1]:.3f} '
                f'| Recall: {rec[-1]:.3f} | Precision: {prec[-1]:.3f}')
            self.res.append([label, classes[label], sum(true_y==label), acc[-1], f1[-1], rec[-1], prec[-1]])
        accuracy = np.mean(true_y == pre_y)
        sum_label = len(true_y)
        P, R, F1 = 0.,0.,0.
        for _,_,size,_,_, rec, pre in self.res:
            R += 1/len(self.res) * rec
            P += 1/len(self.res) * pre
            # print(size, rec, pre)
        F1 = 2*P*R/(P+R)
        print(
            f'\tAverage MixAll | {len(true_y)} | Acc: {accuracy*100:.3f} | F1: {F1:.3f} '
            f'| Recall: {R:.3f} | Precision: {P:.3f}')
        self.res.append([20, 'MixAll', len(true_y), accuracy*100, F1, R, P])
        return acc, f1, rec, prec


    def test(self, test_loader, net, datasetpath):
        # Testing
        epoch_loss = 0.0
        start_time = time.time()
        idx_pre_label = []
        net.eval()
        with torch.no_grad():
            for inputs, key_padding_mask, seq_lens, labels, idx in test_loader:
                inputs = inputs.to(self.args.device)
                key_padding_mask = key_padding_mask.to(self.args.device)
                labels = labels[:, -1].to(self.args.device)
                import torch.nn.functional as F
                # one_hot_labels = F.one_hot(labels[:, -1], num_classes=self.args.num_labels)
                outputs = net(inputs, attention_mask=key_padding_mask, key_padding_mask=key_padding_mask,
                              labels=labels)
                # outputs: [batch_size, seq_len, dim_out] = [256, 200, 20]
                epoch_loss += outputs['loss'].detach()  # 当前批次loss
                pre_y = torch.max(outputs['logits'], dim=1)[1]

                # Save triples of (idx, label, score) in a list
                idx_pre_label += list(zip(idx.to(non_blocking=True).tolist(),
                                                pre_y.to(non_blocking=True).tolist(),
                                                labels.to(non_blocking=True).tolist(),
                                                ))

            # Compute AUC and Loss
            indices, pre_y, labels = zip(*idx_pre_label)
            indices, pre_y, labels = np.array(indices), np.array(pre_y), np.array(labels)

        epoch_loss /= len(test_loader)
        self.test_time = time.time() - start_time

        print(f'\tTest Time:{self.test_time:.3f}s | loss: {epoch_loss:.4f}')

        return self.evaluate(pre_y, labels, datasetpath)


    def load_model(self, model_path, map_location='cuda:0'):
        """Load Deep SAD checkpoints from model_path."""
        with open(f'{model_path}/best_results.json','r') as file:
            best_res = json.load(file)
        model_dict = torch.load(model_path+'/best_checkpoints.tar', map_location=map_location)
        self.args = model_dict['args']
        self.args.target_labels = [10,11,2,4,9]
        if not hasattr(self.args,'beta'):
            self.args.beta = 0.2
        args = self.args
        datasetpath = f'../Dataset/{args.dataset}/Splited_Dataset/'
        self.net = ContrastModel(args=args, contrast_loss=args.use_contrast, graph=args.graph, layer=args.graph_layer, data_path=datasetpath,
                 multi_label=args.mutil, beta=args.beta, alpha=args.alpha, layer_norm_eps=args.layer_norm_eps).to(self.args.device)
        self.net.load_state_dict(model_dict['net_dict'])

    def start(self, model_path='./checkpoints/HMD_main_ContrastModel/'):
        self.load_model(model_path)
        print(model_path)
        set_seed(self.args.seed)
        # self.net.set_criterion(self.args.criterion)
        # print(criterion.named_parameters())
        res_path = './final_test_res/results'
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        timen = ""
        for i in time.localtime()[:6]:
            timen += f"{i}_"

        dataset_path = f'../Dataset/{self.args.dataset}/Splited_Dataset/'

        test_dataset = VeReMi_V2(root=dataset_path, msg_size=self.args.fea_size, type='test')
        test_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                        batch_size=self.args.batch_size,
                                        num_workers=self.args.n_jobs_dataloader, drop_last=False)
        acc, f1, rec, prec = self.test(test_loader, self.net, dataset_path)

        self.res.append([[[i, getattr(self.args, i)] for i in dir(self.args) if not i.startswith('__') and i != 'len']])
        df = pd.DataFrame(data=self.res)
        df.to_csv(res_path + f'/{model_path.split("/")[-2]}_{timen}.csv',  #
                  header=['Type', 'Attack', 'Test_Size', 'Accuracy', 'F1', 'Recall', 'Precision'],
                  index=False, mode='w')  # mode='a':追加
