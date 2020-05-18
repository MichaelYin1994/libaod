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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from utils import LoadSave

np.random.seed(2020)
###############################################################################
def load_mnist(nrows=1000):
    """Loading the MNIST dataset."""
    FILE_NAME = ".//data//mnist//train.csv"
    df = pd.read_csv(FILE_NAME, nrows=nrows)
    X, y = df.drop(["label"], axis=1), df["label"]
    return X, y


def load_fashion_mnist(nrows=1000):
    """Loading the Fashion MNIST dataset."""
    FILE_NAME = ".//data//fashion_mnist//fashion_mnist_train.csv"
    df = pd.read_csv(FILE_NAME, nrows=nrows)
    X, y = df.drop(["label"], axis=1), df["label"]
    return X, y


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
        X, y = total_feats.drop(["target"], axis=1), total_feats["target"]
        return X, y
    else:
        X = total_feats.drop(["target"], axis=1).iloc[:nrows]
        y = total_feats["target"].iloc[:nrows]
        return X, y


def training_clf(clf=None, X_train=None, X_test=None, y_train=None,
                 normalization=False, fillna=False, n_folds=3, stratified=False,
                 shuffle=True, random_state=9102):
    """Training A classifier according to the X_train, validating the performance
    on the validation set, and make predictions on the X_test.

    Parameters
    ----------
    clf : model-like
        A sklearn model with predict_proba method and warm_start attribute.

    X_train : pandas-DataFrame
        A panads DataFrame, the DataFrame will be split into training and 
        validation dataset.

    X_test : pandas-DataFrame
        The testing dataset.

    y_train : Series-like
        The training target.

    y_test : Series-like
        The testing target.

    normalization : TYPE, optional
        If True, the dataset will be normalized first. The default is False.

    fillna : bool
        DESCRIPTION. The default is False.

    n_folds : TYPE, optional
        DESCRIPTION. The default is 3.

    stratified : TYPE, optional
        DESCRIPTION. The default is False.

    shuffle : TYPE, optional
        DESCRIPTION. The default is True.

    random_state : TYPE, optional
        DESCRIPTION. The default is 9102.

    Returns
    -------
    predition results.
    """
    if stratified == True:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle,
                                random_state=random_state)
    else:
        folds = KFold(n_splits=n_folds, shuffle=shuffle,
                      random_state=random_state)

    col_names = list(X_train.columns)
    if fillna:
        for name in col_names:
            X_train[name].fillna(X_train[name].mean(), axis=1, inplace=True)
            X_test[name].fillna(X_test[name].mean(), axis=1, inplace=True)
    if normalization:
        for name in col_names:
            X_train[name] = (
                X_train[name] - X_train[name].mean()) / X_train[name].std()
            X_test[name] = (
                X_test[name] - X_test[name].mean()) / X_test[name].std()

    # Initializing the socre array
    n_classes = len(np.unique(y_train))
    df_score = np.zeros((n_folds, 5))
    oof_pred = np.zeros((len(X_train), n_classes))
    y_test = np.zeros((len(X_train), n_classes))

    # Start training the clf
    print("@Classifier {} started:".format(clf.__name__))
    print("===================================")
    ###########################################################################
    for fold, (tra_id, val_id) in enumerate(folds.split(X_train, y_train)):
        Dtrain, Dval = X_train.loc[tra_id], X_train.loc[val_id]
        Ttrain, Tval = y_train.loc[tra_id], y_train.loc[val_id]

        clf.fit(Dtrain, Ttrain)
        

if __name__ == "__main__":
    # Setting the dataset loading parameter
    dataset_name, nrows = "mnist", None

    # Loading the dataset
    if dataset_name == "mnist":
        X, y = load_mnist(nrows=nrows)
    elif dataset_name == "fashion_mnist":
        X, y = load_fashion_mnist(nrows=nrows)
    elif dataset_name == "boat":
        X, y = load_trajectory(nrows=nrows)
    else:
        raise ValueError("Invalid dataset !")





