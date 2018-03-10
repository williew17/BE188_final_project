# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:06:11 2018

@author: linns
"""

import sklearn.cross_decomposition as skcd
import numpy as np
import csv
from helpers import calc_model_stats

models = [skcd.PLSRegression(n) for n in [3,6,9,12,15,18,21,24,27,30,33,36,39]]

dat_tot=[]
with open('feature_data.csv', newline='') as csvfile:
        tot_data = csv.reader(csvfile, delimiter=',')
        for row in tot_data:
            dat_tot.append(row)
        print('Data read in successfully...')

answers = np.array([np.array((dat_tot[i][1],dat_tot[i+1][1])).flatten() for i in range(0, len(dat_tot)-1, 2)])

data=[np.array((dat_tot[i][2:],dat_tot[i+1][2:])).flatten() for i in range(0,len(dat_tot)-1, 2)]

models2 = [m.fit(data, answers) for m in models]

predictions = [m.predict(data) for m in models2]

for i in range(len(predictions)):
    acc, sens, spec = calc_model_stats(predictions[i], answers)
    for ac, se, sp, name in zip(acc,sens,spec,['Hyperplasic','Serrated','Adenoma']):
        print(name+ ' stats: ')
        print('Accuracy: {0:.2f}%'.format(round(ac*100,2)))
        print('Sensitivity: {0:.2f}%'.format(round(se*100,2)))
        print('Specificity: {0:.2f}%'.format(round(sp*100,2)))
        print('===================')
