#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:55:57 2020

@author: zhuoyin94
"""

import numpy as np
import sklearn.ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, precision_score
from sklearn.metrics import recall_score, log_loss
from base.interfaces import ProbabilisticModel

class RandomForest(ProbabilisticModel):
    def __init__(self, *args, **kwargs):
        self.model = sklearn.ensemble.RandomForestClassifier(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        self.metric = kwargs.pop("metric", None)
        if self.metric not in [None, "f1", "precision", "recall",
                               "log_loss", "roc_auc"]:
            raise ValueError("Invalid metric keyword: {} !".format(
                self.metric))
        if self.metric == None:
            return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

        # Make predictions
        test_pred_proba = self.model.predict_proba(testing_dataset.format_sklearn()[0])
        test_pred_label = np.argmax(test_pred_proba, axis=1)
        _, test_ground_truth = testing_dataset.format_sklearn()

        # Calculation of the scores
        if self.metric == "f1":
            return f1_score(test_ground_truth, test_pred_label, average="macro")
        elif self.metric == "precision":
            return precision_score(test_ground_truth, test_pred_label)
        elif self.metric == "recall":
            return recall_score(test_ground_truth, test_pred_label)
        elif self.metric == "roc_auc":
            enc = OneHotEncoder(sparse=False)
            test_ground_truth_oht = enc.fit_transform(
                test_ground_truth.reshape(-1, 1))
            return roc_auc_score(test_ground_truth_oht, test_pred_proba,
                                 average="macro")
        elif self.metric == "log_loss":
            return log_loss(test_ground_truth, test_pred_label)
        return None

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
