#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:16:26 2020

@author: zhuoyin94
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, auc, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from utils import LoadSave

np.random.seed(2020)
###############################################################################
def load_mnist(nrows=1000):
    """Loading the MNIST dataset."""
    FILE_NAME = ".//data//mnist//train.csv"
    df = pd.read_csv(FILE_NAME, nrows=nrows)
    return df


def load_fashion_mnist(nrows=1000):
    """Loading the Fashion MNIST dataset."""
    FILE_NAME = ".//data//fashion_mnist//fashion_mnist_train.csv"
    df = pd.read_csv(FILE_NAME, nrows=nrows)
    return df


def load_trajectory(nrows=1000):
    """Load the boat trajectory dataset."""
    file_processor = LoadSave(
        ".//data//boat_trajectory//train_feature_lgb.pkl")
    stat_feats = file_processor.load_data()

    embedding_feats = file_processor.load_data(
        path=".//data//boat_trajectory//train_embedding_cbow_list.pkl")[0]
    total_feats = pd.merge(
        stat_feats, embedding_feats, on="boat_id", how="left")

    if nrows == None:
        df = total_feats.drop(["target"], axis=1)
        return df
    else:
        df = total_feats.drop(["target"], axis=1).iloc[:nrows]
        return df


def training_clf(clf=None, train=None, test=None, id_name=None, shuffle=True,
                 target_name=None, fillna=False, normalization=True,
                 random_state=6666):
    """Training A classifier according to the train, validating the performance
    on the test dataset.

    Parameters
    ----------
    clf : model-like
        A sklearn model with predict_proba method and warm_start attribute.

    train : pandas-DataFrame
        A panads DataFrame, the DataFrame will be split into training and 
        validation dataset.

    test : pandas-DataFrame
        The testing dataset. The testing data has the target variable.

    normalization : TYPE, optional
        If True, the dataset will be normalized first. The default is False.

    fillna : bool
        DESCRIPTION. The default is False.

    target_name : str
        The column name in the train and test dataset that can indicate
        which column is the identity column.

    stratified : TYPE, optional
        DESCRIPTION. The default is False.

    shuffle : bool-like, optional
        If true, shuffle the row of the training data first.

    random_state : int, optional
        Random seed. The default is 9102.

    Returns
    -------
    predition results.
    """

    col_names = list(train.columns)
    if shuffle:
        train = train.sample(frac=1)
    if fillna:
        for name in col_names:
            train[name].fillna(train[name].mean(), axis=0, inplace=True)
            test[name].fillna(test[name].mean(), axis=0, inplace=True)
    train_target = train[[id_name, target_name]]
    test_target = test[[id_name, target_name]]
    enc = OneHotEncoder(sparse=False)
    train_target_oht = enc.fit_transform(
        train_target[target_name].values.reshape(-1, 1))
    test_target_oht = enc.fit_transform(
        test_target[target_name].values.reshape(-1, 1))

    if normalization:
        for name in col_names:
            train[name] = (
                train[name] - train[name].mean()) / train[name].std()
            test[name] = (
                test[name] - train[name].mean()) / train[name].std()
    X_train = train.drop([id_name, target_name], axis=1).values
    X_test = test.drop([id_name, target_name], axis=1).values

    clf.fit(X_train, train_target[target_name].values.reshape((-1, 1)))
    train_pred_proba = clf.predict_proba(X_train)
    test_pred_proba = clf.predict_proba(X_test)
    train_pred_label = np.argmax(train_pred_proba, axis=1)
    test_pred_label = np.argmax(test_pred_proba, axis=1)

    train_f1 = f1_score(train_target[target_name].values.reshape((-1, 1)),
                        train_pred_label, average="macro")
    test_f1 = f1_score(test_target[target_name].values.reshape((-1, 1)), 
                       test_pred_label, average="macro")
    train_roc_auc = roc_auc_score(train_target_oht, train_pred_proba,
                                  average="macro")
    test_roc_auc = roc_auc_score(test_target_oht, test_pred_proba,
                                  average="macro")

    print("-- train f1: {:.5f}, roc_auc: {:.5f}, test f1: {:.5f}, roc_auc: {:.5f}\n".format(
        train_f1, train_roc_auc, test_f1, test_roc_auc))
    return clf


if __name__ == "__main__":
    # # Setting the dataset loading parameter
    # dataset_name, nrows = "mnist", None

    # # Loading the dataset
    # if dataset_name == "mnist":
    #     X = load_mnist(nrows=nrows)
    #     X = X.reset_index().rename({"index": "pic_id"}, axis=1)
    # elif dataset_name == "fashion_mnist":
    #     X = load_fashion_mnist(nrows=nrows)
    #     X = X.reset_index().rename({"index": "pic_id"}, axis=1)
    # elif dataset_name == "boat":
    #     X = load_trajectory(nrows=nrows)
    # else:
    #     raise ValueError("Invalid dataset !")

    X_copy = X.copy()
    for col in X_copy.columns:
        if col not in ["pic_id", "label"]:
            X_copy[col] = X_copy[col] / 255
    train_init, test_init = X_copy.iloc[:1000], X_copy.iloc[6000:]

    stride, start = 200, 1000
    batch_dataset = []
    for i in range(20):
        batch_dataset.append(X_copy.iloc[start:(start+stride)])
        start += stride

    # Initial training
    clf = MLPClassifier(hidden_layer_sizes=(50, 20), solver="lbfgs",
                        alpha=5.5, random_state=2090, warm_start=True, tol=1e-2)

    # clf = RandomForestClassifier(n_estimators=300, max_depth=6, 
    #                              n_jobs=-1, warm_start=True)

    # clf = GradientBoostingClassifier(n_estimators=300, max_depth=6,
    #                                  warm_start=True)

    clf = training_clf(clf, train_init, test_init, id_name="pic_id",
                       target_name="label", shuffle=False, fillna=False,
                       normalization=False)

    for ind, dataset in enumerate(batch_dataset):
        print("-- Current batch on {}".format(ind))
        clf = training_clf(clf, dataset, test_init, id_name="pic_id",
                           target_name="label", shuffle=False, fillna=False,
                           normalization=False)
        print("\n")
