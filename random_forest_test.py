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


dat_tot = []
with open('feature_data.csv', newline='') as csvfile:
        tot_data = csv.reader(csvfile, delimiter=',')
        for row in tot_data:
            dat_tot.append(row)
predictions = []
answers = [[dat_tot[i][1],dat_tot[i+1][1]] for i in range(0, len(dat_tot)-1, 2)]
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
     remaining_time = time_passed/(i/2 + 0.0001)*(76-(i/2))
     print ('Approximately {:.2} minutes remaining'.format(remaining_time))
print(predictions)
print(answers)