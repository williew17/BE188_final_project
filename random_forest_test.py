# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:06:11 2018

@author: willie
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import os
from helpers import predict_with_file
from helpers import make_classifiers_predict
from helpers import calc_model_stats

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
acc, sens, spec = calc_model_stats(predictions,answers)
for ac, se, sp, name in zip(acc,sens,spec,['Hyperplasic','Serrated','Adenoma']):
    print(name+ ' stats: ')
    print('Accuracy: {0:.2f}%'.format(round(ac*100,2)))
    print('Sensitivity: {0:.2f}%'.format(round(se*100,2)))
    print('Specificity: {0:.2f}%'.format(round(sp*100,2)))
    print('===================')
