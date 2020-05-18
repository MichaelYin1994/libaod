#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:52:58 2020

@author: zhuoyin94
"""

import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

print(__doc__)

# def load_mnist(nrows=1000):
#     """Loading the MNIST dataset."""
#     FILE_NAME = ".//data//mnist//train.csv"
#     df = pd.read_csv(FILE_NAME, nrows=nrows)
#     X, y = df.drop(["label"], axis=1), df["label"]
#     return X, y

# X, y = load_mnist(nrows=None)
# X = X / 255.

# rescale the data, use the traditional train/test split
X_train, X_test = X[:40000], X[40000:]
y_train, y_test = y[:40000], y[40000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4,
                    solver='lbfgs', verbose=10, random_state=100, tol=1e-10,
                    learning_rate_init=0.01, early_stopping=False)

# this example won't converge because of CI's time constraints, so we catch the
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()