import os
import torch

class Parameters:
    def __init__(self, parameters):
        self.length: int = -1
        self.device = torch.device(f'cuda:{parameters[self.len()]}')  # 使用GPU加速的GpuID，cpu则-1
        self.seed: int = parameters[self.len()]  # 随机种子
        self.n_jobs_dataloader: int = parameters[self.len()]  # dataloader的num_workers
        self.epochs: int = parameters[self.len()]  # 模型训练的迭代次数
        self.learning_rate: float = parameters[self.len()]  # 学习速率/梯度下降的步长
        self.dropout_rate: float = parameters[self.len()]  # dropout的概率一般0.0~0.5
        self.batch_size: int = parameters[self.len()][0]  # 小批量样本数，即每次训练batch_size个
        self.seq_len: int = parameters[self.length][1]  # 单个样本时间序列长度
        self.fea_size: int = parameters[self.length][2]  # 单个时间步的特征数
        self.hidden_size: int = parameters[self.len()]  # Trans的输入的第三维度/模型宽度
        self.dim_feedforward: int = parameters[self.len()]  # Trans的前馈层中间维度
        self.num_labels: int = parameters[self.len()]  # label数
        self.nhead: int = parameters[self.len()]  # 多头自注意力层的头数
        self.num_encoder_layers: int = parameters[self.len()]  # encoder层数
        self.num_decoder_layers: int = parameters[self.len()]  # decoder层数
        self.graph_layer: str = parameters[self.len()]  # graphormer层数
        self.activation: str = parameters[self.len()]  # Trans的激活函数
        self.pooler: str = parameters[self.len()]  # pooler层 'cls' 'mean' 'end'
        self.layer_norm_eps: float = parameters[self.len()]  # layer_norm的标准差分母的add因子
        self.weight_decay: float = parameters[self.len()]  # l2正则化的系数
        self.warm_up: int = parameters[self.len()]  # 学习率warm_up的步数
        self.lr_down: bool = parameters[self.len()]  # 学习率第一阶段下降的步数
        self.use_contrast: bool = parameters[self.len()]  # 是否使用对比学习
        self.mutil: bool = parameters[self.len()]  # 是否为多标签任务
        self.alpha: int = parameters[self.len()]  # graph loss权重
        self.beta: bool = parameters[self.len()]  # 对比学习loss权重
        self.early_stop: int = parameters[self.len()]  # loss N轮不再下降则early_stop
        self.load_ckp: int = parameters[self.len()]  # 是否加载checkpoints
        self.graph: str = parameters[self.len()]    # 是否使用GNN 'GRAPHORMER' or 'GCN' or 'GAT':
        self.criterion: str = parameters[self.len()]  # 'L1Loss'/'MSELoss'/'BCE'/'CE'
        self.model_name: str = parameters[self.len()]  # 模型name
        self.task_name: str = parameters[self.len()]  # 任务name
        self.dataset: str = parameters[self.len()]  # 数据集name
        self.target_labels: list = parameters[self.len()]  # R-drop目标类
        self.model_path = os.getcwd() + '/checkpoints/'  # 模型参数保存的路径

    def len(self):
        self.length += 1
        return self.length

    def __len__(self):
        return self.length
