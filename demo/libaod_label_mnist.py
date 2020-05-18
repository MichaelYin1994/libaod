#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:04:02 2020

@author: zhuoyin94
"""
# import sys
# sys.apth.append("..")
# sys.path.append('/home/zhuoyin94/Desktop/turnout_current_analysis/codes/libaod/libaod')

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from sklearn.model_selection import train_test_split

from libaod.base.dataset import Dataset
from libaod.query_strategies.uncertainty_sampling import UncertaintySampling
from libaod.models import LogisticRegression
from libaod.labelers.interactive_labeler import InteractiveLabeler


def read_raw_csv(nrows=100, dataset="mnist"):
    """Reading raw csv MNIST dataset from .//data_tmp//"""
    if dataset not in ["mnist", "fashion_mnist"]:
        raise ValueError("Invalid dataset name !")

    if dataset == "mnist":
        data = pd.read_csv(".//data_demo//mnist//train.csv", nrows=nrows)
    else:
        data = pd.read_csv(".//data_demo//fashion_mnist//fashion_mnist_train.csv",
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
    data, labels = read_raw_csv(dataset="mnist", nrows=1000)

    # INITIALIZING some parameters
    plot_samples = False
    n_classes = len(np.unique(labels))
    num_batches_run = 1000
    num_need_label_per_batch = 9
    initial_lableded = 200
    error_rate = []

    # Plot randomly 50 digits
    if plot_samples:
        plot_ids = np.random.choice(np.arange(0, len(data)),
                                    size=50, replace=False)
        fig, ax_objs = plt.subplots(3, 3, sharex=True, sharey=True,
                                    figsize=(8, 8))
        ax_objs = ax_objs.ravel()
        for ind, ax in enumerate(ax_objs):
            data_tmp = data[ind, :].reshape((28, 28))
            ax.imshow(data_tmp)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.tight_layout(pad=0.1)
        pause(1)

    # Preparing dataset
    train_data, valid_data = split_train_valid(data=data, labels=labels,
                                                n_labeled=initial_lableded)
    train_data_compare = copy.deepcopy(train_data)

    # Preparing strategy
    qs = UncertaintySampling(train_data, method='lc', n_query_per_batch=num_need_label_per_batch,
                              model=LogisticRegression())

    # Initial error rate
    model = LogisticRegression()
    model.train(train_data)
    error_rate = np.append(error_rate, 1-model.score(valid_data))

    # Give each label its name (labels are from 0 to n_classes-1)
    lbr = InteractiveLabeler(label_name=[str(lbl) for lbl in range(n_classes)],
                             n_query_per_batch=num_need_label_per_batch)

    # Query labeling
    for i in range(num_batches_run):
        '''
        Strategy 1: Uncertainty Sampling
        '''
        ask_id = qs.make_query()

        print("asking sample from Uncertainty Sampling")
        # Plot the Samples need to be labeled
        fig, ax_objs = plt.subplots(
            3, 3, sharex=True, sharey=True, figsize=(8, 8))
        ax_objs = ax_objs.ravel()
        for ax_ind, sa_ind in enumerate(ask_id):
            data_tmp = train_data[sa_ind][0].reshape((28, 28))
            ax_objs[ax_ind].imshow(data_tmp)
            ax_objs[ax_ind].get_xaxis().set_visible(False)
            ax_objs[ax_ind].get_yaxis().set_visible(False)
        fig.tight_layout(pad=0.1)
        plt.show()
        pause(1)

        lb = lbr.label()
        train_data.update(ask_id, lb)
        model.train(train_data)
        error_rate = np.append(error_rate, 1-model.score(valid_data))
