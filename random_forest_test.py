# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:06:11 2018

@author: willi
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import csv
dat_tot = []
with open('feature_data.csv', newline='') as csvfile:
        tot_data = csv.reader(csvfile, delimiter=',')
        for row in tot_data:
            dat_tot.append(row)
y = []; X = []
for sample in dat_tot:
    y.append(sample[1])
    X.append(sample[2:])
clf = RandomForestClassifier(n_estimators = 1000)
clf.fit(X,y)
print(clf.feature_importances_)
print(clf.predict([[dat_tot[0][2:]]]))