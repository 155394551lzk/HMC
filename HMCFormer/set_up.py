import json

import torch
from torch import optim, nn

from HMCFormer.OneCycleLR import OneCycleLR
from HMCFormer.Parameters import Parameters
from HMCFormer.datasets.VeReMi_V2 import VeReMi_V2_Dataset
from HMCFormer.networks.contrast import ContrastModel
from HMCFormer.optim.trainer import Train
from HMCFormer.networks.main import build_network


class HMD(object):
    def __init__(self, args: Parameters = None):

        self.args = args

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.optimizer = None
        self.scheduler = None
        self.trainer = Train(self.args)

        self.results = {
            'train_time': 0.,
            'test_time': 0.,
            'test_acc': 0.,
            'test_rec': 0.,
            'test_pre': 0.,
            'test_f1': 0.
        }

    def set_network(self, net_name, datasetpath):
        """Builds the train network phi."""
        self.net_name = net_name
        args=self.args
        self.net = ContrastModel(args=args, contrast_loss=args.use_contrast, graph=args.graph, layer=args.graph_layer, data_path=datasetpath,
                 multi_label=args.mutil, beta=args.beta, alpha=args.alpha, layer_norm_eps=args.layer_norm_eps)

    def set_optimizer(self, optimizer_name):
        """Builds the optimizer phi."""
        if optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.learning_rate,
                                               betas=(0.9, 0.98), weight_decay=self.args.weight_decay)
        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.learning_rate,
                                              betas=(0.9, 0.98), weight_decay=self.args.weight_decay)
        if optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.learning_rate,
                                             momentum=0.9, weight_decay=0)

    def set_scheduler(self, scheduler_name, total_step):
        """Builds the scheduler phi."""
        if scheduler_name == 'OneCycleLR':
            warm_up = min(self.args.warm_up / (self.args.epochs if self.args.warm_up > 1 else 1), 0.15)
            down = min(self.args.lr_down / (self.args.epochs if self.args.warm_up > 1 else 1), 0.85)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.args.learning_rate * 10, total_steps=total_step,
                                        pct_warmup=warm_up, pct_down=down, anneal_strategy='cos', cycle_momentum=False,
                                        div_factor=10.0, final_div_factor=10.0, three_phase=True, last_epoch=- 1)
        if scheduler_name == 'CyclicLR':
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.args.learning_rate / 25,
                                                         max_lr=self.args.learning_rate,
                                                         step_size_up=total_step // 2,
                                                         step_size_down=total_step // 2, mode="exp_range",
                                                         cycle_momentum=False, gamma=0.999, last_epoch=-1)
        if scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                                  patience=3, verbose=False)
        if scheduler_name == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2, 40, 80, 120], gamma=0.4)
        if scheduler_name == 'CosineAnnealingWarmRestarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=total_step // 30,
                                                                            T_mult=5, eta_min=1,
                                                                            last_epoch=-1)

    def set_criterion(self, criterion_name='MSELoss'):
        """Builds the criterion phi."""
        if criterion_name == 'L1Loss':
            self.criterion = nn.L1Loss(reduction='none')
        if criterion_name == 'MSELoss':
            self.criterion = nn.MSELoss(reduction='none')
        if criterion_name == 'SmoothL1Loss':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        if criterion_name == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        if criterion_name == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()

    def train_And_test(self, dataset: VeReMi_V2_Dataset):
        """Trains the Deep SAD checkpoints on the training data."""
        # Train and test
        if self.optimizer is None:
            self.set_optimizer("AdamW")
        if self.scheduler is None:
            self.set_scheduler("OneCycleLR",
                               int(len(dataset.train_set.targets) / self.args.batch_size) * self.args.epochs)
        self.net = self.trainer.train(dataset, self.net, self.optimizer, self.scheduler)
        self.results['train_time'] = self.trainer.train_time
        # print(f'{self.optimizer.param_groups[0]["lr"]:.6f}')

    def load_model(self, model_path, map_location='cpu'):
        """Load Deep SAD checkpoints from model_path."""
        model_dict = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model_dict['net_dict'])
        self.net = self.net.to(self.args.device)
        print(self.optimizer.state_dict())
        if model_dict['optimizer_state_dict']:
            self.set_optimizer(model_dict['optimizer_state_dict'].__class__.__name__)
            self.optimizer.load_state_dict(model_dict['optimizer_state_dict'].__dict__)
        # self.args = model_dict['args']

    def load_results(self, result_path, map_location='cpu'):
        """load results dict to a JSON-file."""
        result_dict = json.loads(open(result_path).readline())
        self.trainer.train_time = result_dict['train_time']
        self.trainer.test_time = result_dict['test_time']
        self.trainer.best_acc = result_dict['test_acc']
        self.trainer.best_rec = result_dict['test_rec']
        self.trainer.best_pre = result_dict['test_pre']
        self.trainer.best_f1 = result_dict['test_f1']
        # self.args.epochs = result_dict['epoch']
        self.loss = result_dict['loss']
