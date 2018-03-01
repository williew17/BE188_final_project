# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 11:05:31 2018

@author: willie
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import csv
import time
import _pickle as pickle
import os

def predict_with_file(filename, dat_tot, start_time):
    predictions = []
    clf_list = []
    with open(filename, 'rb') as fid:
            clf_list = pickle.load(fid)
    for i in range(len(clf_list)):
        predictions.append(clf_list[i].predict([dat_tot[2*i][2:],dat_tot[2*i+1][2:]]))
        if i%5 == 0:
            time_passed = (time.clock()-start_time)/60
            print (str(int(i)) + " lesions tested in {:.3} minutes".format(time_passed))
            remaining_time = time_passed/(i+1)*(76-i+1)
            print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    return predictions

def make_classifiers_predict(dat_tot, start_time, n_trees = 1000):
    predictions = []
    clf_list = []
    for i in range(0, len(dat_tot)-1, 2):
         dat_tot2 = dat_tot[0:i+1] + dat_tot[i+2:]
         y = []; X = []
         for sample in dat_tot2:
             y.append(sample[1])
             X.append(sample[2:])
         clf = RandomForestClassifier(n_estimators = n_trees)
         clf.fit(X,y)
         clf_list.append(clf)
         predictions.append(clf.predict([dat_tot[i][2:],dat_tot[i+1][2:]]))
         time_passed = (time.clock()-start_time)/60
         print (str(int(i/2)) + " lesions tested in {:.3} minutes".format(time_passed))
         remaining_time = time_passed/((i+1)/2)*(76-((i+1)/2))
         print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    with open('classifiers_trees={}.pkl'.format(n_trees), 'wb') as fid:
        pickle.dump(clf_list, fid)
    return predictions