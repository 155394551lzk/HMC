import os
import sys

from HMCFormer.networks.dtf1 import DecoderLayer, EncoderLayer
from HMCFormer import Parameters
from HMCFormer.networks.other_models import CNN, LSTM, CNN_LSTM, MLP, Mutil_head


def build_network(net_name, args: Parameters = None):
    """Builds the neural network."""

    implemented_networks = (
        'Transformer', 'ContrastModel', 'EncoderLayer', 'DecoderLayer', 'ResNet', 'CNN', 'LSTM', 'CNN_LSTM', 'MLP',
        'Mutil_head')
    assert net_name in implemented_networks

    net = None
    if net_name == 'EncoderLayer':
        net = EncoderLayer(activation=args.activation, seq_len=args.seq_len,
                           fea_size=args.fea_size, num_encoder_layers=args.num_encoder_layers,
                           dropout_rate=args.dropout_rate, nhead=args.nhead,
                           dim_feedforward=args.dim_feedforward, hidden_size=args.hidden_size,
                           layer_norm_eps=args.layer_norm_eps, batch_first=True, device=args.device)

    if net_name == 'DecoderLayer':
        net = DecoderLayer(activation=args.activation, dim_out=args.dim_out, seq_len=args.fea_size,
                           MSG_Size=args.MSG_Size, num_decoder_layers=args.num_decoder_layers,
                           dropout_rate=args.dropout_rate, nhead=args.nhead, batch_size=args.batch_size,
                           dim_feedforward=args.dim_feedforward, hidden_size=args.hidden_size,
                           layer_norm_eps=args.layer_norm_eps, batch_first=True, device=args.device)

    if net_name == 'CNN':
        net = CNN(activation=args.activation, dim_cnn_1=1024, dim_cnn_2=512, kernel_size=[1, 2, 4],
                  seq_len=args.seq_len,
                  dim_out=args.hidden_size)
    if net_name == 'LSTM':
        net = LSTM(args.activation, args.hidden_size, 256, 4, True, args.dropout_rate, args.hidden_size)
    if net_name == 'CNN_LSTM':
        net = CNN_LSTM(activation=args.activation, dim_cnn_1=1024, dim_cnn_2=512, hidden_dim=512,
                       kernel_size=[1, 2, 4], dim_in=args.hidden_size, n_layers=1, bidirectional=True,
                       dropout_rate=args.dropout_rate, dim_out=args.hidden_size)
    if net_name == 'MLP':
        net = MLP(activation=args.activation, MSG_Size=args.MSG_Size, dim_feedforward=args.dim_feedforward,
                  dropout_rate=args.dropout_rate, dim_out=args.dim_out, Seq_Size=args.Seq_Size, device=args.device)
    if net_name == 'Mutil_head':
        net = Mutil_head(activation=args.activation, dim_out=args.dim_out, seq_Size=args.Seq_Size,
                         MSG_Size=args.MSG_Size, num_encoder_layers=args.num_encoder_layers,
                         dropout_rate=args.dropout_rate, nhead=args.nhead,
                         dim_feedforward=args.dim_feedforward,
                         layer_norm_eps=args.layer_norm_eps, batch_first=True, device=args.device)
    return net
