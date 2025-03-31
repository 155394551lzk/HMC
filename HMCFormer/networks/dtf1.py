from torch import nn, optim
import torch
from .optim import _get_activation_fn

from HMCFormer.networks.MutilAtten import MultiheadAttention


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, max_len=20, msg_dim=6, device='cuda:0'):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, msg_dim))
        self.max_len = max_len
        self.device = device
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
            torch.pow(10000, torch.arange(0, msg_dim, 2, dtype=torch.float32) / msg_dim)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self):
        return self.P[:, :self.max_len, :].to(self.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.arg = args
        self.Q = nn.Linear(self.arg.hidden_state, self.arg.hidden_state)
        self.K = nn.Linear(self.arg.hidden_state, self.arg.hidden_state)
        self.V = nn.Linear(self.arg.hidden_state, self.arg.hidden_state)
        self.layer_norm = nn.LayerNorm(self.arg.hidden_state)

        self.head_num = self.arg.head_num

        self.softmax = nn.Softmax(3)

    def forward(self, x, mask):
        cur_batch, seq_len, _ = x.shape
        # copy_x = copy.deepcopy(x)
        copy_x = x

        # mutil-head attn
        q = self.Q(x)
        q = q.reshape(cur_batch, seq_len, self.head_num, -1)
        q = q.transpose(1, 2)

        k = self.K(x)
        k = k.reshape(cur_batch, seq_len, self.head_num, -1)
        k = k.transpose(1, 2)

        v = self.V(x)
        v = v.reshape(cur_batch, seq_len, self.head_num, -1)
        v = v.transpose(1, 2)

        # QK^T
        weight = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(self.arg.hidden_state))
        weight.masked_fill_(mask, -1e9)

        score = self.softmax(weight)

        x = score @ v
        # mutil-head 还原
        x = x.transpose(1, 2).reshape(cur_batch, seq_len, -1)

        return x


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""

    def __init__(self, num_encoder_layers=12, dim_in=128, dim_feedforward=512, activation='relu', dim_out=128):
        super(PositionWiseFFN, self).__init__()
        self.activation = _get_activation_fn(activation)
        self.dense1 = nn.Linear(dim_in, dim_feedforward, bias=False)
        self.dense2 = nn.Linear(dim_feedforward, dim_out, bias=False)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight, gain=self.Beta)

    def forward(self, X):
        return self.dense2(self.activation(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, layer_norm_eps=1e-5, num_layers=12, dropout_rate=0.1, hidden_size=128):
        super(AddNorm, self).__init__()
        self.Alpha = (2 * num_layers) ** 0.25
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))  # self.dropout(Y)  *self.Alpha


class DecoderBlock(nn.Module):
    """解码器中第i个块"""

    def __init__(self, num_layers=12, dim_out=2, hidden_size=128, nhead=8, dim_feedforward=512,
                 activation='relu', layer_norm_eps=1e-5, dropout_rate=0.1, batch_first=True, i=1, device='cuda:0',
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.device = device
        self.attention1 = MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout_rate, kdim=hidden_size,
                                             vdim=hidden_size, batch_first=batch_first)
        self.addnorm1 = AddNorm(num_layers=num_layers, layer_norm_eps=layer_norm_eps,
                                dropout_rate=dropout_rate, hidden_size=hidden_size)
        self.ffn = PositionWiseFFN(num_encoder_layers=num_layers, dim_in=hidden_size,
                                   dim_feedforward=dim_feedforward,
                                   activation=activation, dim_out=hidden_size)
        self.addnorm2 = AddNorm(num_layers=num_layers, layer_norm_eps=layer_norm_eps,
                                dropout_rate=dropout_rate, hidden_size=hidden_size)

    def forward(self, X, key_padding_mask, attn_mask):
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        if self.training:
            # 包含单向mask和padding_mask的自注意力
            Y = self.addnorm1(X, self.attention1(X, X, X, key_padding_mask=key_padding_mask,
                                                 attn_mask=attn_mask)[0])
        else:
            # 只包含padding_mask的自注意力
            Y = self.addnorm1(X, self.attention1(X, X, X, key_padding_mask=key_padding_mask)[0])

        return self.addnorm2(Y, self.ffn(Y))


class DecoderLayer(nn.Module):
    def __init__(self, activation="relu", seq_len=200, fea_size=10, num_decoder_layers=12,
                 dropout_rate=0.1, nhead=8, dim_feedforward=512, hidden_size=128, dim_out=20, layer_norm_eps=1e-6,
                 batch_first=True, device='cuda:0', **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.device = device
        self.seq_len = seq_len
        self.embedding = nn.Linear(fea_size, hidden_size, bias=False)  # 对输入进行深层次的嵌入
        self.pos_code1 = PositionalEncoding(seq_len, fea_size, device).forward()  # 绝对位置编码
        self.pos_code2 = nn.Embedding(seq_len, fea_size, device=device)  # 可学习位置编码
        atten_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        self.atten_mask = atten_mask.float().masked_fill(atten_mask == 0, bool(1)).\
            masked_fill(atten_mask == 1, bool(0)).to(self.device)
        self.blks = nn.Sequential()
        for i in range(num_decoder_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(num_layers=num_decoder_layers, dim_out=hidden_size, hidden_size=hidden_size,
                                              nhead=nhead, dim_feedforward=dim_feedforward,
                                              activation=activation, layer_norm_eps=layer_norm_eps,
                                              dropout_rate=dropout_rate, batch_first=batch_first, i=i, device=device, ))
        self._attention_weights = [None] * len(self.blks)
        # self.dense = nn.Linear(hidden_size, dim_out)
        self.register_buffer("position_ids", torch.arange(seq_len).expand((1, -1)))

    def init_state(self, *args):
        return [None] * self.num_layers

    def forward(self, inputs, key_padding_mask, embedding_weight=None, position_ids=None, past_key_values_length=0):
        # if embedding_weight is not None:
        #     if embedding_weight.ndim == 2:
        #         embedding_weight = embedding_weight.unsqueeze(-1)
        #     inputs *= embedding_weight
        # if position_ids is None:
        #     position_ids = self.position_ids[:, past_key_values_length: self.seq_len + past_key_values_length]
        # input_with_pos = inputs + self.pos_code2(position_ids)
        # input_emds = self.embedding(input_with_pos)
        outputs = inputs
        # pos_idx = torch.arange(0, X.size(1)).view(1, -1).expand(X.size(0), X.size(1)).to(X.device)
        # X = self.embedding(X + self.pos_code2(pos_idx))
        # atten_mask: [batch_size,attn_head_num, seq_length, seq_length] 每一个头都为相同的下三角矩阵
        for i, blk in enumerate(self.blks):
            outputs = blk(outputs, key_padding_mask, self.atten_mask)
            # 解码器自注意力权重
            # self._attention_weights[i] = blk.attention1.attention.attention_weights
        # logits = self.dense(X)
        return outputs

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderBlock(nn.Module):
    """transformer编码器块"""

    def __init__(self, num_encoder_layers=12, dim_out=2, hidden_size=128, nhead=8, dim_feedforward=512,
                 activation='relu', layer_norm_eps=1e-5, dropout_rate=0.1, batch_first=True):
        super(EncoderBlock, self).__init__()
        self.Beta = (8 * num_encoder_layers) ** (-0.25)
        self.attention = MultiheadAttention(embed_dim=hidden_size, num_heads=nhead, dropout=dropout_rate, kdim=hidden_size,
                                            vdim=hidden_size, batch_first=batch_first)
        self.addnorm1 = AddNorm(num_layers=num_encoder_layers, layer_norm_eps=layer_norm_eps,
                                dropout_rate=dropout_rate, hidden_size=hidden_size)
        self.ffn = PositionWiseFFN(num_encoder_layers=num_encoder_layers, dim_in=hidden_size,
                                   dim_feedforward=dim_feedforward,
                                   activation=activation, dim_out=hidden_size)
        self.addnorm2 = AddNorm(num_layers=num_encoder_layers, layer_norm_eps=layer_norm_eps,
                                dropout_rate=dropout_rate, hidden_size=hidden_size)

        for name, param in self.attention.named_parameters():
            if 'weight' in name:
                if name.startswith(("v_proj", "out_proj")):
                    nn.init.xavier_normal_(param, gain=self.Beta)
                elif name.startswith(("q_proj", "k_proj")):
                    # print(name, param)
                    nn.init.xavier_normal_(param, gain=1)

    def forward(self, X, key_padding_mask):
        Y = self.addnorm1(X, self.attention(X, X, X, key_padding_mask=key_padding_mask)[0])
        return self.addnorm2(Y, self.ffn(Y))


class EncoderLayer(nn.Module):
    """transformer编码器块
    input:[batch,seq_Size,MSG_size]
    output:[batch,hidden_size]
    """

    def __init__(self, activation="relu", seq_len=20, fea_size=6, num_encoder_layers=12, dropout_rate=0.1,
                 nhead=8, dim_feedforward=512, hidden_size=128, dim_out=2, layer_norm_eps=1e-5, batch_first=True,
                 device='cuda:0'):
        super(EncoderLayer, self).__init__()
        # self.seq_len = seq_len
        # self.activation = _get_activation_fn(activation)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.pos_code1 = PositionalEncoding(seq_len, fea_size, device).forward()  # 绝对位置编码
        # self.pos_code2 = nn.Embedding(seq_len, fea_size, device=device)  # 可学习位置编码
        # self.embedding = nn.Linear(fea_size, hidden_size, bias=False)  # 对输入进行深层次的嵌入
        self.blks = nn.Sequential()
        for i in range(num_encoder_layers):
            self.blks.add_module("block" + str(i), EncoderBlock(num_encoder_layers=num_encoder_layers, dim_out=hidden_size,
                                                                hidden_size=hidden_size, nhead=nhead,
                                                                dim_feedforward=dim_feedforward,
                                                                activation=activation, layer_norm_eps=layer_norm_eps,
                                                                dropout_rate=dropout_rate, batch_first=batch_first))
        # self.ffn_out = PositionWiseFFN(num_encoder_layers, hidden_size, dim_feedforward, activation, dim_out)
        # self.register_buffer("position_ids", torch.arange(seq_len).expand((1, -1)))

    def forward(self, input_emds, key_padding_mask, embedding_weight=None, position_ids=None, past_key_values_length=0):
        # if embedding_weight is not None:
        #     if embedding_weight.ndim == 2:
        #         embedding_weight = embedding_weight.unsqueeze(-1)
        #     inputs *= embedding_weight
        # if position_ids is None:
        #     position_ids = self.position_ids[:, past_key_values_length: self.seq_len + past_key_values_length]
        # input_with_pos = inputs + self.pos_code2(position_ids)
        # input_emds = self.embedding(input_with_pos)
        outputs = input_emds
        for i, blk in enumerate(self.blks):
            outputs = blk(outputs, key_padding_mask)
        # logits = self.dense(X)
        # outputs = self.ffn_out(outputs)
        return outputs