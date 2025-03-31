import os
import random

import numpy as np
from torch.backends import cudnn
import torch
from HMCFormer.Parameters import Parameters
from HMCFormer.set_up import HMD
from HMCFormer.datasets.VeReMi_V2 import VeReMi_V2_Dataset
import sys


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True  # 保证CNN的可复现性
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def main(ARGS):
    if ARGS.task_name != '':
        ARGS.model_path += ARGS.task_name + '/'
    else:
        ARGS.model_path += str(os.path.basename(sys.argv[0][:-3]) + '_' + ARGS.model_name + '/')
    # make modelpath
    if not os.path.exists(ARGS.model_path):
        os.mkdir(ARGS.model_path)

    set_seed(ARGS.seed)
    # 1.加载dataset
    # 数据集调用：dataset.(train_set/test_set/...).(data/data/targets)
    datasetpath = f'../Dataset/{ARGS.dataset}/Splited_Dataset/'
    print(datasetpath)
    dataset = VeReMi_V2_Dataset(root=datasetpath, msg_size=ARGS.fea_size)
    print(f"Train:Evaluate = {len(dataset.train_set.targets)}:{len(dataset.test_set.targets)} "
          f"\nTrain: Type_size = {dataset.train_set.type_size}"
          f"\nEval: Type_size = {dataset.test_set.type_size}")
    # label_dict = {i: v for i, v in enumerate(label_dict)}
    ARGS.seq_len, ARGS.fea_size = dataset.train_set.max_len, dataset.train_set.fea_size
    ARGS.num_labels = dataset.n_classes
    # ARGS.num_labels = len(label_dict)

    # 2. 模型定义
    model = HMD(args=ARGS)
    model.set_network('ContrastModel', datasetpath)  # net: encoder
    # 3. 定义优化器optimizer、学习率策略scheduler，必须在2.之后，因为定义优化器必须用到net的参数
    model.set_optimizer('AdamW')
    model.set_criterion(ARGS.criterion)
    model.set_scheduler('OneCycleLR', int(len(dataset.train_set.targets) / ARGS.batch_size) * (ARGS.epochs + 1))

    # 4. 模型和结果加载
    if ARGS.load_ckp and os.path.exists(ARGS.model_path + 'best_checkpoints.tar'):
        model.load_model(model_path=ARGS.model_path + 'best_checkpoints.tar', map_location=ARGS.device)
        print('checkpoints loaded.')
        if os.path.exists(ARGS.model_path + 'best_results.json'):
            model.load_results(result_path=ARGS.model_path + 'best_results.json', map_location=ARGS.device)
            print('results loaded.')

    for i in dir(model.args):
        if not i.startswith('__') and i != 'len':
            print(f"{i}:{getattr(model.args, i)}")

    # 5. 模型训练和测试
    model.train_And_test(dataset)

    print(f'Time:{model.trainer.train_time:.3f}s ,Best result: '
          f'\nEpoch:{model.trainer.best_epoch} | Acc: {model.trainer.best_acc * 100:.3f} | F1: '
          f'{model.trainer.best_f1 * 100:.3f} | Recall:{model.trainer.best_rec * 100:.3f} | '
          f'Precision: {model.trainer.best_pre * 100:.3f}')

    from HMCFormer.optim.Final_test import Final_test
    final_test = Final_test().start(model_path=ARGS.model_path)


if __name__ == "__main__":
    # for checkpoints in ['MLP']: GRAPHORMER [10, 11, 2, 4, 9]
    # ('Transformer', 'EncoderLayer', 'ResNet', 'CNN', 'LSTM', 'CNN_LSTM', 'MLP')
    print("ce+rdrop Starting")
    main(Parameters([0, 1, 4, 400, 1e-4, 0.1, [300, 200, 10], 128, 512, 20, 4, 6, 6, 6, 'relu', 'cls', 1e-7, 1e-3,
         0.02, 0.3, True, False, 1., 0.2, 6, False, '', 'BCE', 'EncoderLayer', f'VeReMi_ce+graph',
                     'VeReMi',[1, 2, 5]] ))
    print("ce+rdrop Ended")

    print("ce+rdrop+graph Starting")
    main(Parameters([0, 1, 4, 400, 1e-4, 0.1, [300, 200, 10], 128, 512, 20, 4, 6, 6, 6, 'relu', 'cls', 1e-7, 1e-3,
         0.02, 0.3, True, False, 1., 0.2, 6, False, 'GRAPHORMER', 'BCE', 'EncoderLayer', f'VeReMi_ce+graph',
                     'VeReMi',[1, 2, 5]] ))
    print("ce+rdrop+graph Ended")