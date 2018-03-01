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
import time
import _pickle as pickle
import os

dat_tot = []
with open('feature_data.csv', newline='') as csvfile:
        tot_data = csv.reader(csvfile, delimiter=',')
        for row in tot_data:
            dat_tot.append(row)
predictions = []
<<<<<<< HEAD
answers = np.array([np.array([dat_tot[i][1],dat_tot[i+1][1]]) for i in range(0, len(dat_tot)-1, 2)])
start_time = time.clock()
for i in range(0, len(dat_tot)-1, 2):
     dat_tot2 = dat_tot[0:i+1] + dat_tot[i+2:]
     y = []; X = []
     for sample in dat_tot2:
         y.append(sample[1])
         X.append(sample[2:])
     clf = RandomForestClassifier(n_estimators = 1000)
     clf.fit(X,y)
     predictions.append(clf.predict([dat_tot[i][2:],dat_tot[i+1][2:]]))
     time_passed = (time.clock()-start_time)/60
     print (str(i/2) + " lesions tested in {:.3} minutes".format(time_passed))
     remaining_time = time_passed/(i/2 + 0.01)*(76-(i/2))
     print ('Approximately {:.2} minutes remaining'.format(remaining_time))
predictions = np.array(predictions)
print(predictions)
print(answers)

comparison = np.equal(predictions, answers)
print(comparison)
comparison = np.swapaxes(comparison,0,1)
#this is the number of matches
comparison = [np.sum([1 if b else 0 for b in a]) for a in comparison]
print(comparison)
=======
answers = [[dat_tot[i][1],dat_tot[i+1][1]] for i in range(0, len(dat_tot)-1, 2)]
if os.path.isfile('classifiers_trees={}.pkl'.format(100)):
    clf_list = []
    with open('classifiers_trees={}.pkl'.format(100), 'rb') as fid:
            clf_list = pickle.load(fid)
    for i in range(len(clf_list)):
        predictions.append(clf_list[i].predict([dat_tot[2*i][2:],dat_tot[2*i+1][2:]]))
else:
    start_time = time.clock()
    clf_list = []
    for i in range(0, len(dat_tot)-1, 2):
         dat_tot2 = dat_tot[0:i+1] + dat_tot[i+2:]
         y = []; X = []
         for sample in dat_tot2:
             y.append(sample[1])
             X.append(sample[2:])
         clf = RandomForestClassifier(n_estimators = 1000)
         clf.fit(X,y)
         clf_list.append(clf)
         predictions.append(clf.predict([dat_tot[i][2:],dat_tot[i+1][2:]]))
         time_passed = (time.clock()-start_time)/60
         print (str(i/2) + " lesions tested in {:.3} minutes".format(time_passed))
         remaining_time = time_passed/(i/2 + 0.0001)*(76-(i/2))
         print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    with open('classifiers_trees={}.pkl'.format(100), 'wb') as fid:
        pickle.dump(clf_list, fid)
print(predictions)
>>>>>>> ebd86865873a5df6b6378a5f61fae4df333d0a5a
