#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202005201035
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994
# Reference:  https://github.com/ntucllab/libact

"""
此模块（data.py）包含了libaod的基础接口类，包括：
-- 基础数据类
-- 图像数据类
"""

from __future__ import unicode_literals
import numpy as np

class Dataset:
    """libaod的基础数据类。

    @Parameters:
    ----------
    data : {array-like or list-like}, shape = (n_samples, )
        Raw data of sample set, the value can be empty.

    feat : {array-like}, shape = (n_samples, n_features)
        Features of sample set, features can be used to training a machine
        learning model.

    y : list of {int, None} or array-like, shape = (n_samples)
        The ground truth (label) for corresponding sample. Unlabeled data
        should be given a label None.

    @Attributes:
    ----------
    data : list, shape = (n_samples)
        List of all sample feature and label tuple.
    """
    def __init__(self, data=None, feat=None, y=None):
        if data is None:
            data = np.array([])
        if not isinstance(data, np.ndarray) and not isinstance(data, list):
            raise TypeError("Invalid data type: {} !".format(type(data)))

        if feat is None:
            feat = np.array([])
        if not isinstance(feat, np.ndarray):
            raise TypeError("Invalid feat type: {} !".format(type(feat)))

        if y is None:
            y = []
        if not isinstance(y, list) and not isinstance(y, np.ndarray):
            raise TypeError("Invalid y type: {} !".format(type(y)))
        y = np.array(y)

        self._data = data
        self._feat = feat
        self._y = y
        self.modified = True
        self._update_callback = set()

    def __len__(self):
        return self._feat.shape[0]

    def __getitem__(self, idx):
        return self._data[idx], self._feat[idx], self._y[idx]

    @property
    def data(self): return self

    def get_labeled_mask(self):
        return ~np.fromiter((e is None for e in self._y), dtype=bool)

    def len_labeled(self):
        return self.get_labeled_mask().sum()

    def len_unlabeled(self):
        return (~self.get_labeled_mask()).sum()

    def get_num_of_labels(self):
        return np.unique(self._y[self.get_labeled_mask()]).size

    def append(self, feat, label=None):
        if isinstance(self._feat, np.ndarray):
            self._feat = np.vstack([self._feat, feat])
        self._y = np.append(self._y, label)

        self.modified = True
        return len(self) - 1

    def on_update(self, callback):
        """
        Add callback function to call when dataset updated.

        Parameters
        ----------
        callback : callable
            The function to be called when dataset is updated.
        """
        self._update_callback.add(callback)

    def update(self, entry_id, new_label):
        """
        Updates an entry with entry_id with the given label

        Parameters
        ----------
        entry_id : int
            entry id of the sample to update.

        label : {int, None}
            Label of the sample to be update.
        """
        self._y[entry_id] = new_label
        self.modified = True
        for callback in self._update_callback:
            callback(entry_id, new_label)

    def format_sklearn(self):
        # becomes the same as get_labled_entries
        feat, labels = self.get_labeled_feat_labels()
        return feat, np.array(labels)

    def get_feat_labels(self):
        return self._feat, self._y

    def get_labeled_feat_labels(self):
        labeled_mask = self.get_labeled_mask()
        return self._feat[labeled_mask], self._y[labeled_mask].tolist()

    def get_unlabeled_feat_labels(self):
        unlabeled_mask = ~self.get_labeled_mask()
        return self._feat[unlabeled_mask], self._y[unlabeled_mask].tolist()

    def get_data_labels(self):
        return self._data, self._y.tolist()

    def get_labeled_data_labels(self):
        labeled_mask = self.get_labeled_mask()
        if isinstance(self._data, list):
            ret_data, ret_labels = [], []
            for ind, item in enumerate(labeled_mask):
                if item == True:
                    ret_data.append(self._data[ind])
                    ret_labels.append(self._y[ind])
                else:
                    break
        elif isinstance(self._data, np.ndarray):
            ret_data = self._data[labeled_mask]
            ret_labels = self._y[labeled_mask]
        return ret_data, ret_labels

    def get_unlabeled_data_labels(self):
        unlabeled_mask = ~self.get_labeled_mask()
        if isinstance(self._data, list):
            ret_data, ret_labels = [], []
            for ind, item in enumerate(unlabeled_mask):
                if item == True:
                    continue
                else:
                    ret_data.append(self._data[ind])
                    ret_labels.append(self._y[ind])
        elif isinstance(self._data, np.ndarray):
            ret_data = self._data[unlabeled_mask]
            ret_labels = self._y[unlabeled_mask]
        return ret_data, ret_labels

    def get_unlabeled_feat_ids(self):
        return np.where(~self.get_labeled_mask())[0], self._feat[~self.get_labeled_mask()]

