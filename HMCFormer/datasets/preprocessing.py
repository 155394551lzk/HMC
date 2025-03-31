import torch
import numpy as np


def create_semisupervised_setting(labels, normal_classes, outlier_classes, ratio_known_normal, ratio_known_outlier):
    """
    Create a semi-supervised data setting. 
    """
    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()  # 正常数据的下标
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()  # 异常数据的下标
    n_normal = len(idx_normal)  # 正常数据的数目
    n_anormal = len(idx_outlier)  # 异常数据的数目
    # print(ratio_known_normal, ratio_known_outlier, ratio_pollution)
    print("idx_normal, idx_outlier:", len(idx_normal), len(idx_outlier))

    # Get number of samples
    n_known_normal = int(n_normal * ratio_known_normal)
    n_unlabeled_normal = n_normal - n_known_normal
    n_known_outlier = int(ratio_known_outlier * n_anormal)
    n_unlabeled_outlier = n_anormal - n_known_outlier

    print('n_known_normal,n_unlabeled_normal, n_unlabeled_outlier, n_known_outlier: '
          , n_known_normal, n_unlabeled_normal, n_unlabeled_outlier, n_known_outlier)

    # Get original class labels

    # Get semi-supervised setting labels, outlier:-1, normal:1, unknown:0
    semi_labels = np.zeros(n_normal+n_anormal).astype(np.int32).tolist()
    for i in range(n_known_normal):
        semi_labels[idx_normal[i]] = 1
    for i in range(n_known_outlier):
        semi_labels[idx_outlier[i]] = -1

    return semi_labels
