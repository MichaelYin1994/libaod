#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:50:38 2020

@author: zhuoyin94
"""

# import sys
# sys.path.append("..")
# sys.path.append('/home/zhuoyin94/Desktop/turnout_current_analysis/codes/libaod/libaod')

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
    feat.drop(["device_id", "record_id"], axis=1, inplace=True)
    data.drop(["device_id", "record_id", "date_time", "target"],
              axis=1, inplace=True)
    data = [[data["phase_a"].iloc[i], data["phase_b"].iloc[i],
             data["phase_c"].iloc[i]] for i in range(len(data))]
    return data, feat, labels


def split_train_valid(data=None, labels=None, n_labeled=1000):
    """Split the training and testing data."""
    n_classes = 2
    # X_train, X_valid, y_train, y_valid = train_test_split(data, labels,
    #                                                       test_size=0.4)
    # while len(np.unique(y_train[:n_labeled])) < n_classes:
    #     X_train, X_valid, y_train, y_valid = train_test_split(data, labels,
    #                                                           test_size=0.4)

    train_data = Dataset(X_train, X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train)-n_labeled)]))
    valid_data = Dataset(X_valid, X_valid, y_valid)
    return train_data, valid_data


if __name__ == "__main__":
    data, feat, labels = read_data_feats()