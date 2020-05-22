#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:50:38 2020

@author: zhuoyin94
"""

import sys
sys.path.append("..")
sys.path.append('/home/zhuoyin94/Desktop/turnout_current_analysis/codes/libaod/libaod')

import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from sklearn.model_selection import train_test_split

# from libaod.base.dataset import Dataset
from libaod.base.data import Dataset
from libaod.query_strategies.uncertainty_sampling import UncertaintySampling
from libaod.models import RandomForest
from libaod.labelers.interactive_labeler import InteractiveLabeler
from utils import LoadSave

def read_data_feats(nrows=100):
    """Reading raw turnout dataset from .//data_demo//"""
    file_processor = LoadSave()
    data = file_processor.load_data(
        path=".//data_demo//turnout//samples_data.pkl")
    feat = file_processor.load_data(
        path=".//data_demo//turnout//samples_feat.pkl")

    
    labels = feat["target"].values
    feat.drop(["device_id", "record_id", "date_time"], axis=1, inplace=True)
    data.drop(["device_id", "record_id", "date_time", "target"],
              axis=1, inplace=True)
    data = [[data["phase_a"].iloc[i], data["phase_b"].iloc[i],
             data["phase_c"].iloc[i]] for i in range(len(data))]
    return data, feat, labels


def split_train_valid(data=None, feat=None, labels=None, n_labeled=1000):
    """Split the training and testing data."""
    n_trains = int(0.5*len(data))
    idx_array = np.arange(0, len(data))
    np.random.shuffle(idx_array)
    train_idx, valid_idx = idx_array[:n_trains], idx_array[n_trains:]

    # Split the training and testing data.
    train_data, vaild_data = [], []
    for i in train_idx:
        train_data.append(data[i])
    for i in valid_idx:
        vaild_data.append(data[i])

    train_feat, valid_feat = feat.iloc[train_idx].values, feat.iloc[valid_idx].values
    train_label, valid_label = labels.iloc[train_idx].values, labels.iloc[valid_idx].values

    # Dataset object
    train_data = Dataset(feat=train_feat, data=train_data, y=np.concatenate(
        [train_label[:n_labeled], [None] * (len(train_label)-n_labeled)]))
    valid_data = Dataset(data=vaild_data, feat=valid_feat, y=valid_label)
    return train_data, valid_data


if __name__ == "__main__":
    data, feat, labels = read_data_feats()

    # INITIALIZING some parameters
    n_classes = len(np.unique(labels))
    num_batches_run = 500
    num_need_label_per_batch = 9
    initial_lableded = 12000
    test_scores = []

    train_data, valid_data = split_train_valid(data=data,
                                               labels=feat["target"],
                                               n_labeled=initial_lableded,
                                               feat=feat.drop("target", axis=1))

    # Initial error rate
    model = RandomForest(n_estimators=200, max_features="sqrt", n_jobs=-1)
    model.train(train_data)
    test_scores = np.append(test_scores, model.score(valid_data, metric="f1"))

    # Preparing strategy
    qs = UncertaintySampling(train_data, method='lc', n_query_per_batch=num_need_label_per_batch,
                             model=RandomForest(n_estimators=200, max_features="sqrt", n_jobs=-1))

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
        fig, ax_objs = plt.subplots(3, 3, figsize=(8, 5))
        ax_objs = ax_objs.ravel()
        for ax_ind, sa_ind in enumerate(ask_id):
            data_tmp = train_data[sa_ind][0]
            ax_objs[ax_ind].plot(data_tmp[0], color="r", lw=2)
            ax_objs[ax_ind].plot(data_tmp[1], color="g", lw=2)
            ax_objs[ax_ind].plot(data_tmp[2], color="b", lw=2)
            ax_objs[ax_ind].tick_params(axis="y", labelsize=8)
            ax_objs[ax_ind].tick_params(axis="x", labelsize=8)
            ax_objs[ax_ind].set_xlim(0, len(data_tmp[0]))
            ax_objs[ax_ind].set_ylim(0, 7)
            ax_objs[ax_ind].grid(True)

        fig.tight_layout(pad=0.1)
        plt.show()
        pause(1)

        lb = lbr.label()
        train_data.update(ask_id, lb)
        model.train(train_data)
        test_scores = np.append(test_scores, model.score(valid_data, metric="f1"))
