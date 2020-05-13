#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:36:56 2020

@author: zhuoyin94
"""

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling, RandomSampling
from libact.labelers import InteractiveLabeler


def read_raw_csv(nrows=100, dataset="mnist"):
    """Reading raw csv MNIST dataset from .//data_tmp//"""
    if dataset not in ["mnist", "fashion_mnist"]:
        raise ValueError("Invalid dataset name !")

    if dataset == "mnist":
        data = pd.read_csv(".//data_tmp//mnist//train.csv", nrows=nrows)
    else:
        data = pd.read_csv(".//data_tmp//fashion_mnist//fashion_mnist_train.csv",
                           nrows=nrows)
    pix_col_names = [name for name in data.columns if "pix" in name]
    labels = data["label"].values
    data = data[pix_col_names].values

    return data, labels


def split_train_valid(data=None, labels=None, n_labeled=100):
    """Split the training and testing data."""
    n_classes = len(np.unique(labels))
    X_train, X_valid, y_train, y_valid = train_test_split(data, labels,
                                                          test_size=0.4)
    while len(np.unique(y_train[:n_labeled])) < n_classes:
        X_train, X_valid, y_train, y_valid = train_test_split(data, labels,
                                                              test_size=0.4)

    train_data = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train)-n_labeled)]))
    valid_data = Dataset(X_valid, y_valid)
    return train_data, valid_data


if __name__ == "__main__":
    data, labels = read_raw_csv(dataset="fashion_mnist", nrows=None)
    data, labels = read_raw_csv(dataset="mnist", nrows=None)

    # INITIALIZING some parameters
    plot_samples = False
    n_classes = len(np.unique(labels))
    num_batches_run = 1000
    num_need_label_per_batch = 10
    initial_lableded = 1500
    error_rate_0, error_rate_1 = [], []

    # # Plot randomly 50 digits
    # if plot_samples:
    #     plot_ids = np.random.choice(np.arange(0, len(data)),
    #                                 size=50, replace=False)
    #     fig, ax_objs = plt.subplots(5, 10, sharex=True, sharey=True,
    #                                 figsize=(16, 8))
    #     ax_objs = ax_objs.ravel()
    #     for ind, ax in enumerate(ax_objs):
    #         data_tmp = data[ind, :].reshape((28, 28))
    #         ax.imshow(data_tmp)
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #     fig.tight_layout(pad=0.1)
    #     pause(1)

    # # Preparing dataset
    # train_data, valid_data = split_train_valid(data=data, labels=labels,
    #                                             n_labeled=initial_lableded)
    # train_data_compare = copy.deepcopy(train_data)

    # # Preparing strategy
    # qs = UncertaintySampling(train_data, method='lc',
    #                          model=LogisticRegression())
    # qs2 = RandomSampling(train_data_compare)


    # # Initial error rate
    # model = LogisticRegression()
    # model.train(train_data)
    # error_rate_0 = np.append(error_rate_0, 1-model.score(valid_data))

    # model.train(train_data_compare)
    # error_rate_1 = np.append(error_rate_1, 1-model.score(valid_data))

    # # Give each label its name (labels are from 0 to n_classes-1)
    # lbr = InteractiveLabeler(label_name=[str(lbl) for lbl in range(n_classes)])

    # # Query labeling
    # for i in range(num_batches_run):
    #     '''
    #     Strategy 1: Uncertainty Sampling
    #     '''
    #     ask_id = qs.make_query()
    #     print("asking sample from Uncertainty Sampling")
    #     tmp_0 = train_data.data[ask_id][0].reshape(28, 28)

    #     plt.imshow(tmp_0)
    #     pause(1)
    #     qs_uncertain_sampling = train_data.data[ask_id][0].reshape(28, 28)

    #     lb = lbr.label(train_data.data[ask_id][0].reshape(28, 28))
    #     train_data.update(ask_id, lb)
    #     model.train(train_data)
    #     error_rate_0 = np.append(error_rate_0, 1-model.score(valid_data))

    #     '''
    #     Strategy 2: Random Sampling
    #     '''
    #     ask_id = qs2.make_query()
    #     print("asking sample from Uncertainty Sampling")
    #     tmp_1 = train_data_compare.data[ask_id][0].reshape(28, 28)
    #     plt.imshow(tmp_1)
    #     pause(1)
    #     qs_random_sampling = train_data_compare.data[ask_id][0].reshape(28, 28)

    #     lb = lbr.label(train_data_compare.data[ask_id][0].reshape(28, 28))
    #     train_data_compare.update(ask_id, lb)
    #     model.train(train_data_compare)
    #     error_rate_1 = np.append(error_rate_1, 1 - model.score(valid_data))
