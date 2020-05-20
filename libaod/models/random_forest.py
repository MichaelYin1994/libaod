#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:55:57 2020

@author: zhuoyin94
"""

import numpy as np
import sklearn.ensemble
from base.interfaces import ProbabilisticModel

class RandomForest(ProbabilisticModel):
    def __init__(self, *args, **kwargs):
        self.model = sklearn.ensemble.RandomForestClassifier(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
