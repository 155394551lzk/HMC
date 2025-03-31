import json
import math
import os

import pandas as pd
from matplotlib import pyplot as plt

from HMCFormer.base.base_net import BaseNet
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import time
import torch
import numpy as np
from HMCFormer.Parameters import Parameters


class Train():

    def __init__(self, args: Parameters = None):
        super(Train, self).__init__()
        self.train_lr = []
        self.train_loss = []
        self.ce_loss = []
        self.graph_loss = []
        self.rdrop_loss = []
        self.args = args
        # Optimization parameters
        self.eps = args.layer_norm_eps

        # Results
        self.train_time = 0.
        self.test_time = 0.
        self.best_acc = 0.
        self.best_rec = 0.
        self.best_pre = 0.
        self.best_f1 = 0.
        self.best_epoch = 0

    def evaluate(self, pre_y, true_y, average='macro'):
        # pre_y = torch.tensor(pre_y)

        # true_y = torch.tensor(true_y)
        # for i in range(len(true_y)):
        #     print("真实标签，预测标签，MAE：", true_y[i], pre_y_all[i], mae[i])
        # print(len(pre_y_all), len(mae))
        # 计算F1,rec,prec,acc
        # features = set(true_y)
        # classes = [line.strip() for line in open("./AckType.txt").readlines()]
        # for label in features:
        #     tp = np.sum((true_y == label) & (pre_y == label))
        #     tn = np.sum((true_y != label) & (pre_y != label))
        #     fp = np.sum((true_y != label) & (pre_y == label))
        #     fn = np.sum((true_y == label) & (pre_y != label))
        #     # p = tp/(tp+fp) r = tp/(tp+fn)
        #     f1 = (2 * tp / (2 * tp + fp + fn + self.args.layer_norm_eps))
        #     rec = (tp / (tp + fn + self.args.layer_norm_eps))
        #     prec = (tp / (tp + fp + self.args.layer_norm_eps))
        #     acc = ((tp + tn) / (tp + tn + fn + fp))
        #     # print(f"\t 预测标签{i}：{(pre_y_all == i).sum()} 预测正确:预测错误={tp}:{fp}")
        #     print(
        #         f'\tLabel:{classes[label]} \t Acc: {acc * 100:.3f} | F1: {f1 * 100:.3f} '
        #         f'| Recall: {rec * 100:.3f} | Precision: {prec * 100:.3f}')
        accuracy = np.mean(true_y == pre_y)
        precision, recall, f1_score, _ = precision_recall_fscore_support(true_y, pre_y, average=average)

        return accuracy, recall, precision, f1_score

    def train(self, dataset, net: BaseNet, optimizer, scheduler):
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.args.batch_size, num_workers=self.args.n_jobs_dataloader)
        net = net.to(self.args.device)
        early_stop_count = 0
        # Training
        start_time = time.time()
        print(f'Training Start')
        for epoch in range(self.args.epochs):
            epoch_loss, ce_loss, graph_loss, rdrop_loss = 0.0, 0.0, 0.0, 0.0
            epoch_start_time = time.time()
            net.train()
            if early_stop_count >= self.args.early_stop:
                print("Early stop!")
                break

            # Train
            for inputs, key_padding_mask, _, labels, _ in train_loader:
                # Zero the network parameter gradients
                optimizer.zero_grad(set_to_none=True)
                inputs = inputs.to(self.args.device)
                key_padding_mask = key_padding_mask.to(self.args.device)
                labels = labels[:, -1].to(self.args.device)
                import torch.nn.functional as F
                # one_hot_labels = F.one_hot(labels[:, -1], num_classes=self.args.num_labels)
                outputs = net(inputs, attention_mask=key_padding_mask, key_padding_mask=key_padding_mask,
                              labels=labels)

                net.zero_grad(set_to_none=True)  # 梯度归零
                outputs['loss'].backward()  # 梯度下降
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # 防止梯度爆炸
                optimizer.step()  # 参数修改
                epoch_loss += outputs['loss'].detach()  # 当前批次loss
                ce_loss += outputs['ce_loss']
                graph_loss += outputs['graph_loss']
                rdrop_loss += outputs['rdrop_loss']
                scheduler.step()

            # log epoch statistics
            train_loss = epoch_loss / len(train_loader)
            epoch_train_time = time.time() - epoch_start_time
            print(
                f'\tEpochs {epoch + 1:02}:Train Time:{epoch_train_time:.3f}s | loss:{train_loss:.5f} | '
                f'ce_loss:{ce_loss / len(train_loader):.5f} | graph_loss:{graph_loss / len(train_loader):.5f} | '
                f'contrastive_loss:{rdrop_loss / len(train_loader):.5f} | '
                f'lr: {optimizer.param_groups[0]["lr"]:.6f}')
            self.train_lr.append(optimizer.param_groups[0]["lr"])
            self.train_loss.append(float(train_loss))
            self.ce_loss.append(float(ce_loss/len(train_loader)))
            self.graph_loss.append(float(graph_loss / len(train_loader)))
            self.rdrop_loss.append(float(rdrop_loss / len(train_loader)))
            # self.save_model(export_model=self.args.model_path + f'{epoch}_checkpoints.tar',net=net, optimizer=optimizer, epoch=epoch)
            # use dataset to test the encoder_net
            if epoch > self.args.epochs * 0.6 // 1:
                self.test(dataset, net, epoch, optimizer)

        self.save_lr_loss()
        self.train_time = time.time() - start_time
        return net

    def test(self, dataset, net: BaseNet, epoch, optimizer):
        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.args.batch_size, num_workers=self.args.n_jobs_dataloader)
        # Testing
        epoch_loss = 0.0
        start_time = time.time()
        net.eval()
        idx_pre_label_score = []
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
                idx_pre_label_score += list(zip(idx.to(non_blocking=True).tolist(),
                                                pre_y.to(non_blocking=True).tolist(),
                                                labels.to(non_blocking=True).tolist(),
                                                ))

            # Compute AUC and Loss
            indices, pre_y, labels = zip(*idx_pre_label_score)
            indices, pre_y, labels = np.array(indices), np.array(pre_y), np.array(labels)

        epoch_loss /= len(test_loader)
        # print results
        # evaluate Acc/F1/Rec/Pre
        acc, f1, recall, prec = self.evaluate(pre_y, labels)

        # 如果acc更好，且其他三个中有两个更好，则视为有更好的结果
        if acc > self.best_acc and (
                int(f1 > self.best_f1) + int(recall > self.best_rec) + int(prec > self.best_pre)) > 1:
            self.best_acc = acc
            self.best_f1 = f1
            self.best_rec = recall
            self.best_pre = prec
            self.best_epoch = epoch + 1
            self.save_model_result(net, optimizer, epoch, epoch_loss, True)
        self.test_time = time.time() - start_time

        print(f'\tTest Time:{self.test_time:.3f}s | loss: {epoch_loss:.4f} | Acc: {acc * 100:.3f} | '
              f'F1: {f1 * 100:.3f} | Recall: {recall * 100:.3f} | Precision: {prec * 100:.3f}')

        print(
            f'\tBest result: | Epoch:{self.best_epoch} | Acc: {self.best_acc * 100:.3f} | F1: {self.best_f1 * 100:.3f} '
            f'| Recall:{self.best_rec * 100:.3f} | Precision: {self.best_pre * 100:.3f}')

        # print('Second stage: test ends.')

    def save_model_result(self, net: BaseNet, optimizer, epoch, epoch_loss, best_epoch=False):
        """Tests the Deep SAD checkpoints on the test data."""
        # Get results
        self.results = {
            'train_time': self.train_time,
            'test_time': self.test_time,
            'test_acc': self.best_acc,
            'test_rec': self.best_rec,
            'test_pre': self.best_pre,
            'test_f1': self.best_f1,
            'epoch': epoch,
            'loss': epoch_loss.item()
        }
        # save result
        if not os.path.exists(self.args.model_path):
            os.mkdir(self.args.model_path)
        if best_epoch:
            self.save_results(export_json=self.args.model_path + f'best_results.json')
            self.save_model(export_model=self.args.model_path + f'best_checkpoints.tar',
                            net=net, optimizer=optimizer, epoch=epoch)
        else:
            self.save_results(export_json=self.args.model_path + f'{epoch}_results.json')
            self.save_model(export_model=self.args.model_path + f'{epoch}_checkpoints.tar',
                            net=net, optimizer=optimizer, epoch=epoch)

    def save_lr_loss(self):
        df = pd.DataFrame()
        df['train_lr'] = self.train_lr
        df['train_loss'] = self.train_loss
        df['ce_loss'] = self.ce_loss
        df['graph_loss'] = self.graph_loss
        df['rdrop_loss'] = self.rdrop_loss
        timen = ""
        for i in time.localtime()[:6]:
            timen += f"{i}_"
        df.to_csv(f'./final_test_res/lr_loss/{self.args.model_path.split("/")[-2]}_{timen}_lrloss.csv',
                  index=False, mode='w')  # mode='a':追加
        print('lr_score saved')

    def save_model(self, export_model, net: BaseNet = None, optimizer=None, epoch=0):
        """Save Deep SAD checkpoints to export_model."""
        net_dict = net.state_dict()
        torch.save({'net_dict': net_dict,
                    'optimizer_state_dict': optimizer,
                    'args': self.args,
                    'epoch': epoch
                    }, export_model)

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def plt_lr_loss(self):
        plt.plot(self.train_lr)
        plt.title(f"lr")
        plt.show()

        plt.plot(self.train_loss)
        plt.title(f"loss")
        plt.show()

        plt.plot(self.train_lr, self.train_loss)
        plt.title(f"lr-loss")
        plt.xlabel(f'lr')
        plt.ylabel('loss')
        plt.show()
