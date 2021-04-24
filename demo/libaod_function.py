#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:57:22 2020

@author: zhuoyin94
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(2020)
###############################################################################
def read_data_feats():
    """Reading raw turnout dataset from .//data_demo// for testing."""
    with open(".//data_demo//turnout//samples_data.pkl", 'rb') as file:
        data = pickle.load(file)
    with open(".//data_demo//turnout//samples_feat.pkl", 'rb') as file:
        feat = pickle.load(file)

    labels = feat["target"].values
    feat.drop(["device_id", "record_id", "date_time"], axis=1, inplace=True)
    data.drop(["device_id", "record_id", "date_time", "target"],
              axis=1, inplace=True)
    data = [[data["phase_a"].iloc[i], data["phase_b"].iloc[i],
             data["phase_c"].iloc[i]] for i in range(len(data))]

    feat["device_id"] = list(np.arange(len(feat), 0, -1))
    feat["record_id"] = list(np.arange(0, len(feat)))
    return data, feat, labels


def random_forest_training(train_data=None, train_label=None, params=None):
    """Random Forest Quick Training."""
    if params is None:
        params = {"n_estimators": 200,
                  "max_depth": 6,
                  "n_jobs": -1,
                  "oob_score": True}

    clf = RandomForestClassifier(**params)
    clf.fit(train_data, train_label)
    return clf


def active_generating_queries(feat_table=None, **kwargs):
    """
    ----------
    Author: Michael Yin
    E-Mail: zhuoyin94@163.com
    ----------
    
    @Description:
    ----------
        Active label generating according to the classifier predition 
        probability.

    @Parameters:
    ----------
    feat_table: {pandas DataFrame-like}
        The DataFrame that is used train the Machine Learning Algorithm.
        The shape is like:
           phase_a_mean  phase_b_mean  ... target
        0      1.281014      1.347609           0
        1      1.540133      1.472467           1
        2      1.615455      1.628601           0
        3      1.487794      1.388824           None
        4      1.282245      1.399252           None
        In this DataFrame, the labeled instances have labels with 0/1, and
        unlabeled instances have labels with None feature.

    **kwargs: {dict-like}
        params: {dict-like}
            Training params for a sklearn classifier.
        method: {str-like}
            Query startegy, including "least confident", "entropy".
        n_query_per_batch: {int-like}, default 10
            # queries per batch.

    @Return:
    ----------
    Query results.
    """
    if not isinstance(feat_table, pd.DataFrame):
        raise TypeError("Invalid feat_table type ! The feat_table type is {}, the required type is DataFrame !".format(
            type(feat_table)))
    feat_cols = list(feat_table.columns)

    if "target" not in feat_cols:
        raise ValueError("No target column in feat_table !")
    if "device_id" not in feat_cols or "record_id" not in feat_cols:
        raise ValueError("No identity columns in feat_table !")

    unique_labels = feat_table["target"].unique()
    for label in unique_labels:
        if label not in [0, 1] and (np.isnan(label) == False):
            raise ValueError("Invalid target label: {}".format(label))

    # Pre-set parameters
    params = kwargs.pop("params", None)
    method = kwargs.pop("method", "lc")
    n_query_per_batch = kwargs.pop("n_query_per_batch", 10)
    drop_list = ["device_id", "record_id", "target", "date_time"]

    # Data tables
    feat_labeled = feat_table[~feat_table["target"].isnull()]
    feat_unlabeled = feat_table[feat_table["target"].isnull()]

    # Accessing the trained classifier
    clf = random_forest_training(feat_labeled.drop(drop_list, axis=1),
                                 feat_labeled["target"].values, params)

    # Predicting on the testing data
    test_pred_proba = clf.predict_proba(feat_unlabeled.drop(drop_list, axis=1))
    test_normal_id = np.argsort(test_pred_proba[:, 0])[::-1][:n_query_per_batch]
    test_abnormal_id = np.argsort(test_pred_proba[:, 1])[::-1][:n_query_per_batch]

    # Query generation
    ret_query_res, ret_normal_res, ret_abnormal_res = None, None, None
    if method == "lc":
        test_pred_proba = -np.max(test_pred_proba, axis=1)
    elif method == "entropy":
        test_pred_proba = np.sum(
            -test_pred_proba * np.log(test_pred_proba), axis=1)

    query_id = np.argsort(test_pred_proba)[::-1][:n_query_per_batch]

    ret_query_res = feat_unlabeled[["device_id", "record_id"]].iloc[query_id]
    ret_normal_res = feat_unlabeled[["device_id", "record_id"]].iloc[test_normal_id]
    ret_abnormal_res = feat_unlabeled[["device_id", "record_id"]].iloc[test_abnormal_id]

    return ret_query_res, ret_normal_res, ret_abnormal_res


if __name__ == "__main__":
    data, feat, labels = read_data_feats()
    feat["date_time"] = None
    feat_with_nan = feat.copy()
    feat_with_nan["target"].iloc[15000:] = None

    tmp = active_generating_queries(feat_with_nan, method="entropy")
