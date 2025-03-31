
from torch import nn, optim
import torch

from HMCFormer.networks.MutilAtten import MultiheadAttention
from HMCFormer.networks.dtf1 import PositionalEncoding
import torch.nn.functional as F

class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, num_encoder_layers=12, dim_in=128, dim_feedforward=512, activation='relu', dim_out=128):
        super(PositionWiseFFN, self).__init__()
        self.activation = _get_activation_fn(activation)
        self.Beta = (8 * num_encoder_layers) ** (-0.25)
        self.dense1 = nn.Linear(dim_in, dim_feedforward, bias=False)
        self.dense2 = nn.Linear(dim_feedforward, dim_out, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=self.Beta)

    def forward(self, X):
        return self.dense2(self.activation(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, layer_norm_eps=1e-5, num_encoder_layers=12, dropout_rate=0.1, d_model=128):
        super(AddNorm, self).__init__()
        self.Alpha = (2 * num_encoder_layers) ** 0.25
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, X, Y):
        return self.ln(X * self.Alpha + self.dropout(Y))  # self.dropout(Y)


class CNN(nn.Module):
    def __init__(self, activation='relu', dim_cnn_1=1024, dim_cnn_2=512, kernel_size=[1, 2, 4], seq_len=200,
                 dim_in=128, dim_out=128):
        super(CNN, self).__init__()  # 继承__init__功能
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.activation = _get_activation_fn(activation)
        self.dim_out = dim_out
        self.CNN = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim_in, dim_cnn_1, (Size,), bias=False),
                    # nn.BatchNorm1d(dim_cnn_1, momentum=0.5, affine=True),
                    # self.activation,
                    nn.Conv1d(dim_cnn_1, dim_cnn_2, (Size,), bias=False),
                    nn.BatchNorm1d(dim_cnn_2, momentum=0.5, affine=True),
                    self.activation,
                )
                for Size in self.kernel_size
            ]
        )
        # self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.fc = nn.Linear(dim_cnn_2 * len(self.kernel_size), dim_out)
        # self.max_pool1d = nn.MaxPool1d()
        self.ln = nn.LayerNorm(dim_out)

    def forward(self, batch_x, key_padding_mask):
        batch_x = batch_x.permute(0, 2, 1)
        # print(batch_x.size())
        cnn_out = [conv(batch_x).squeeze(2) for conv in self.CNN]
        cnn_out = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in cnn_out]
        # print(len(cnn_out), cnn_out[0].size(), cnn_out[1].size(), cnn_out[2].size())
        cnn_out = torch.cat(cnn_out, dim=1).squeeze(-1)
        x = self.ln(self.fc(cnn_out))
        return x.unsqueeze(1)


class LSTM(nn.Module):
    def __init__(self, activation, dim_in, hidden_dim, n_layers, bidirectional, dropout_rate, dim_out):
        super(LSTM, self).__init__()  # 继承__init__功能
        self.activation = _get_activation_fn(activation)
        self.bidirectional = bidirectional
        self.dim_out = dim_out
        self.LSTM = nn.LSTM(input_size=dim_in,  # 输入的特征维度，也就是特征长度被卷积后的长度
                            hidden_size=hidden_dim,  # 隐藏层节点个数
                            num_layers=n_layers,  # 隐藏层层数
                            batch_first=True,  # 输入中第一维为batch_size
                            bidirectional=bidirectional,  # 双向LSTM
                            dropout=dropout_rate)
        self.fc = nn.Linear(int(self.bidirectional + 1) * n_layers * hidden_dim, dim_out)
        self.ln = nn.LayerNorm(dim_out)

    def forward(self, batch_x, key_padding_mask):
        # print(f'原始：{batch_x.size()}')  # batch_x:[batch, 200, 10]
        # output = torch.zeros(batch_x.size(0), self.seq_len, self.dim_out).to(batch_x.device)
        out, (h_n, _) = self.LSTM(batch_x)  # out: [batch, 200, num_directions * hidden_dim]
        out = h_n.permute(1, 0, 2).reshape(batch_x.size(0), -1)
        # print(out.size())
        out = self.ln(self.fc(out))
        return out.unsqueeze(1)

# class CNN2(nn.Module):
#     def __init__(self, activation='relu', dim_cnn_1=1024, dim_cnn_2=512, kernel_size=[1, 2, 4], seq_len=200,
#                  dim_out=128):
#         super(CNN2, self).__init__()  # 继承__init__功能
#         self.kernel_size = kernel_size
#         self.seq_len = seq_len
#         self.activation = _get_activation_fn(activation)
#         self.dim_out = dim_out
#         self.CNN = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv1d(seq_len, dim_cnn_1, (Size,), bias=False),
#                     # nn.BatchNorm1d(dim_cnn_1, momentum=0.5, affine=True),
#                     # self.activation,
#                     nn.Conv1d(dim_cnn_1, dim_cnn_2, (Size,), bias=False),
#                     nn.BatchNorm1d(dim_cnn_2, momentum=0.5, affine=True),
#                     self.activation,
#                 )
#                 for Size in kernel_size
#             ]
#         )
#         self.fc = nn.Linear(dim_cnn_2 * len(self.kernel_size), dim_out)
#         self.ln = nn.LayerNorm(dim_out)
#
#     def forward(self, batch_x, key_padding_mask):
#         cnn_out = [conv(batch_x) for conv in self.CNN]
#         cnn_out = torch.cat(cnn_out, dim=1).squeeze(-1)
#         x = self.ln(self.fc(cnn_out))
#         return x.unsqueeze(1)

class CNN_LSTM(nn.Module):
    def __init__(self, activation, dim_cnn_1=1024, dim_cnn_2=512, hidden_dim: int = 512, n_layers: int = 2,
                 bidirectional: bool = True, dropout_rate: float = 0.0, kernel_size: list = None, dim_in: int = 128,
                 dim_out: int = 128):
        super(CNN_LSTM, self).__init__()  # 继承__init__功能
        self.CNN = CNN(activation, dim_cnn_1, dim_cnn_2, kernel_size, dim_out)
        self.LSTM = LSTM(activation, dim_in, hidden_dim, n_layers, bidirectional, dropout_rate, dim_out)

    def forward(self, batch_x, key_padding_mask):
        return self.LSTM(self.CNN(batch_x, key_padding_mask), key_padding_mask)


# MLP构建
class MLP(nn.Module):
    def __init__(self, activation, Seq_Size, MSG_Size, device, dim_feedforward, dropout_rate, dim_out):
        super(MLP, self).__init__()  # 继承__init__功能
        self.activation = _get_activation_fn(activation)
        self.MSG_Size = MSG_Size
        self.poscode = PositionalEncoding(Seq_Size, MSG_Size, device).forward()
        self.ffn = PositionWiseFFN(activation=activation, dim_feedforward=dim_feedforward,
                                   dim_in=128, dim_out=dim_out)
        self.embedding = nn.Linear(MSG_Size, 128, bias=True)  # 对输入进行深层次的嵌入
        self.ln = nn.LayerNorm(128)
        self.bn = nn.BatchNorm1d(Seq_Size)

    def forward(self, X, key_padding_mask):
        src = self.activation(self.ln(self.embedding((X + self.poscode))))
        return self.ffn(src)


class Encoder_Ablation(nn.Module):
    def __init__(self, num_encoder_layers=12, dim_out=2, d_model=128, nhead=8, dim_feedforward=512,
                 activation='relu', layer_norm_eps=1e-5, dropout_rate=0.1, batch_first=True):
        super(Encoder_Ablation, self).__init__()  # 继承__init__功能
        self.Beta = (8 * num_encoder_layers) ** (-0.25)
        self.attention = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate, kdim=d_model,
                                            vdim=d_model, batch_first=batch_first)
        self.addnorm1 = AddNorm(num_encoder_layers=num_encoder_layers, layer_norm_eps=layer_norm_eps,
                                dropout_rate=dropout_rate, d_model=d_model)
        self.ffn = PositionWiseFFN(num_encoder_layers=num_encoder_layers, dim_in=d_model,
                                   dim_feedforward=dim_feedforward,
                                   activation=activation, dim_out=dim_out)
        self.addnorm2 = AddNorm(num_encoder_layers=num_encoder_layers, layer_norm_eps=layer_norm_eps,
                                dropout_rate=dropout_rate, d_model=d_model)

    def forward(self, X, key_padding_mask):
        Y = self.addnorm1(X, self.attention(X, X, X, key_padding_mask=key_padding_mask)[0])
        return self.addnorm2(Y, self.ffn(Y))
        # return self.ffn(batch_x)


class Mutil_head(nn.Module):
    def __init__(self, activation="relu", seq_Size=20, MSG_Size=6, num_encoder_layers=12, dropout_rate=0.1,
                 nhead=8, dim_feedforward=512, d_model=128, dim_out=2, layer_norm_eps=1e-5, batch_first=True,
                 device='cuda:0'):
        super(Mutil_head, self).__init__()  # 继承__init__功能
        self.Beta = (8 * num_encoder_layers) ** (-0.25)
        self.poscode = PositionalEncoding(seq_Size, MSG_Size, device).forward()
        self.embedding = nn.Linear(MSG_Size, d_model, bias=False)
        self.attention = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate, kdim=d_model,
                                            vdim=d_model, batch_first=batch_first)
        self.ffn = PositionWiseFFN(num_encoder_layers=num_encoder_layers, dim_in=d_model,
                                   dim_feedforward=dim_feedforward,
                                   activation=activation, dim_out=dim_out)

    def forward(self, X, key_padding_mask):
        X = self.embedding(X + self.poscode)
        Y = self.attention(X, X, X, key_padding_mask=key_padding_mask)[0]
        return self.ffn(Y)
        # return self.ffn(batch_x)


def _get_activation_fn(activation):
    if activation.upper() == "RELU":
        return nn.ReLU()
    elif activation.upper() == "GLU":
        return nn.GLU()
    elif activation.upper() == "SWISH":
        return nn.Hardswish()
    elif activation.upper() == "LEAKYRELU":
        return nn.LeakyReLU()
    elif activation.upper() == "RRELU":
        return nn.RReLU()
    elif activation.upper() == "ELU":
        return nn.ELU()
    elif activation.upper() == "PRELU":
        return nn.PReLU()
    elif activation.upper() == "MISH":
        return nn.Mish()
    elif activation.upper() == "SELU":
        return nn.SELU()
    else:
        return nn.GELU()
