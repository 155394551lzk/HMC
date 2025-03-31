import math

from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .graph import GraphEncoder
from .main import build_network
from .optim import _get_activation_fn
from HMCFormer.networks.dtf1 import EncoderLayer, DecoderLayer
from HMCFormer.Parameters import Parameters


class PoolingLayer(nn.Module):
    def __init__(self, avg='mean'):
        super(PoolingLayer, self).__init__()
        self.avg = avg

    def forward(self, x):
        if self.avg == 'end':
            x = x[:, -1, :]
        elif self.avg == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return x


class NTXent(nn.Module):

    def __init__(self, args: Parameters, tau=1.):
        super(NTXent, self).__init__()
        self.tau = tau
        self.norm = 1.
        self.transform = nn.Sequential(
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.hidden_size, args.hidden_size),
        )

    def forward(self, x, labels=None):
        x = self.transform(x)
        n = x.shape[0]
        x = F.normalize(x, p=2, dim=1) / np.sqrt(self.tau)
        # 2B * 2B
        sim = x @ x.t()
        sim[np.arange(n), np.arange(n)] = -1e9

        logprob = F.log_softmax(sim, dim=1)

        m = 2

        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm

        return loss

    # def forward(self, embeddings, temperature=1.0):
    #     embeddings = self.transform(embeddings)
    #     embeddings = F.normalize(embeddings, p=2, dim=1) / np.sqrt(self.tau)
    #     # 计算相似度矩阵
    #     similarity_matrix = embeddings @ embeddings.T / temperature
    #
    #     # 构建标签，相同样本对的标签为1，不同样本对的标签为0
    #     labels = torch.arange(similarity_matrix.size(0)).to(embeddings.device)
    #
    #     # 计算NT-Xent损失
    #     loss = F.cross_entropy(similarity_matrix, labels)
    #
    #     return loss


class ContrastModel(nn.Module):
    def __init__(self, args: Parameters, contrast_loss=True, graph='', layer=6, data_path=None,
                 multi_label=False, beta=0.2, alpha=1, layer_norm_eps=1e-6):
        super(ContrastModel, self).__init__()
        self.target_labels = args.target_labels
        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(args.hidden_size, args.num_labels)
        self.model = build_network(args.model_name, args)
        self.pooler = PoolingLayer('cls')
        self.contrast_loss = contrast_loss
        self.graph = graph
        if graph != '':
            self.graph_encoder = GraphEncoder(args, graph, layer=layer, data_path=data_path)
        self.beta = beta
        self.alpha = alpha
        self.seq_len = args.seq_len

        # self.init_weights()

        self.multi_label = multi_label
        self.pos_code2 = nn.Embedding(args.seq_len, args.fea_size, device=args.device)  # 可学习位置编码
        self.embedding = nn.Linear(args.fea_size, args.hidden_size, bias=False)  # 对输入进行深层次的嵌入
        self.register_buffer("position_ids", torch.arange(args.seq_len).expand((1, -1)))

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def forward(
            self,
            original_seq=None,
            attention_mask=None,
            key_padding_mask=None,
            labels=None,
            past_key_values=None,
    ):
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        position_ids = self.position_ids[:, past_key_values_length: self.seq_len + past_key_values_length]
        # original_seq [seq_len, fea_size]
        input_with_pos = original_seq + self.pos_code2(position_ids)
        input_emds = self.embedding(input_with_pos)
        seq_outputs = self.model(input_emds, key_padding_mask)
        pooled_output = self.dropout(self.pooler(seq_outputs))
        logits = self.classifier(pooled_output)

        ce_loss, graph_loss, rdrop_loss = 0., 0., 0.
        if attention_mask is None:
            attention_mask = torch.ones(((original_seq.size(0), original_seq.size(1) + past_key_values_length)),
                                        device=original_seq.device)
        if labels is not None:
            if self.multi_label:
                loss_fct = nn.BCEWithLogitsLoss()
                target = labels.float()
            else:
                loss_fct = CrossEntropyLoss()
                target = labels

            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                ce_loss = loss_fct(logits.view(-1), target.view(-1))
            else:
                ce_loss = loss_fct(logits, target)

            if self.training:
                if self.graph:
                    # print(input_emds.size())
                    embedding_weight = self.graph_encoder(input_emds, attention_mask, labels)
                    weighted_original_seq = original_seq * embedding_weight + self.pos_code2(position_ids)
                    weighted_input_emds = self.embedding(weighted_original_seq)
                    graph_seq_outputs = self.model(weighted_input_emds, key_padding_mask)
                    graph_seq_output = self.dropout(self.pooler(graph_seq_outputs))
                    graph_logits = self.classifier(graph_seq_output)
                    graph_loss = loss_fct(graph_logits, target)

                if self.contrast_loss:
                    selected_indices = torch.isin(target, torch.tensor(self.target_labels).to(target.device))
                    if selected_indices.any():
                        # 使用布尔索引从标签张量中选择对应的样本标签
                        selected_input_emds = input_emds[selected_indices]
                        selected_key_padding_mask = key_padding_mask[selected_indices]
                        contrast_seq_outputs = self.model(selected_input_emds, selected_key_padding_mask)
                        contrast_seq_outputs = self.dropout(self.pooler(contrast_seq_outputs))
                        contrast_logits = self.classifier(contrast_seq_outputs)
                        ce_loss += loss_fct(contrast_logits, target[selected_indices])
                        rdrop_loss = self.compute_kl_loss(logits[selected_indices], contrast_logits)

        loss = ce_loss + self.alpha * graph_loss + self.beta * rdrop_loss

        if ce_loss > 0:
            ce_loss = ce_loss.item()
        if graph_loss > 0:
            graph_loss = graph_loss.item()
        if rdrop_loss > 0:
            rdrop_loss = rdrop_loss.item()
        return {
            'loss': loss,
            'ce_loss': ce_loss,
            'graph_loss': graph_loss,
            'rdrop_loss': rdrop_loss,
            'logits': logits,
            'seq_outputs': seq_outputs,
        }
