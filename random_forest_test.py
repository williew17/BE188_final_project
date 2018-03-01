# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:06:11 2018

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
from helpers import predict_with_file
from helpers import make_classifiers_predict

dat_tot = []
with open('feature_data.csv', newline='') as csvfile:
        tot_data = csv.reader(csvfile, delimiter=',')
        for row in tot_data:
            dat_tot.append(row)
        print('Data read in successfully...')
predictions = []
answers = np.array([np.array([dat_tot[i][1],dat_tot[i+1][1]]) for i in range(0, len(dat_tot)-1, 2)])
start_time = time.clock()
clf_list = []
if os.path.isfile('classifiers_trees={}.pkl'.format(1000)):
    predictions = predict_with_file('classifiers_trees={}.pkl'.format(1000), dat_tot, start_time)
else:
    predictions = make_classifiers_predict(dat_tot, start_time, n_trees = 1000)
predictions = np.array(predictions)
print(predictions.shape)
print(answers.shape)
'''        
comparison = np.equal(np.array(predictions), answers)
print(comparison)
comparison = np.swapaxes(comparison,0,1)
#this is the number of matches
comparison = [np.sum([1 if b else 0 for b in a]) for a in comparison]
print(comparison)
'''