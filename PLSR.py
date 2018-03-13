# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:06:11 2018

@author: linus
"""

import sklearn.cross_decomposition as skcd
import numpy as np
import csv
import argparse
from helpers import *

    def check_output_by_twos(array,axis):
    '''Use powers of two to see if multiple classifications exist for a datapoint.
    Input: an n-d list of booleans, the axis to compress over 
    Output: a list combined over the axis, where the 1st True adds 1, the 2nd True adds 2, etc.'''

    if axis > 0:
            yield from check_output_by_twos(array, axis-1)
    else:
        try:
            iter(array)
            for a in len(array):
                yield check_output_by_twos(array[a], axis-1, a)
        except TypeError:
                yield 2**a if array else 0

parser = argparse.ArgumentParser()
parser.add_argument('--cmode', type = str, default = 'binary', help = 'chooses binary vs multi classification mode')
parser.add_argument('--folds', type = int, default = -1, help = 'chooses the amount folds for cross validation')
FLAGS = parser.parse_args()

models = [skcd.PLSRegression(n) for n in [3,30]]
classes = ['Hyp (b)', 'Ser (m)', 'Ade (m)']

feature_data=[]
with open('feature_data.csv', newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        for row in data_reader:
            feature_data.append([int(row[1])] + [float(r) for r in row[2:]])
        print('Data read in successfully...')

#multiclass prediction is 3 runs of PLSR-DA, for x vs. not-x
#for binary, only check the first run of hyp vs. not-hyp
#pair indicates that the 2 rows were combined into one
pair_modified_answers = []
pair_data = [feature_data[i][1:] + feature_data[i+1][1:] for i in range(0, len(feature_data)-1, 2)]
pair_predictions = []

for i in range(len(classes)):
        pair_modified_answers.append([1 if feature_data[j][0] == (i+1) else -1 for j in range(0,len(feature_data)-1, 2)])
for c in range(len(classes)):
    pp=[]
    for m in models:
        m.fit(pair_data, pair_modified_answers[c])
        pp.append(np.squeeze(m.predict(pair_data)))
    pair_predictions.append(pp)

print(pair_predictions[0])
#convert numbers (floats) to classes(ints) from cutoff of 0
cutoff=0
pair_predicted_classes = [[[True if ppp > cutoff else False for ppp in pp] for pp in p] for p in pair_predictions]

print(check_output_by_twos(pair_predicted_classes, 0))

#for i in range(len(predictions)):
#    acc, sens, spec = multi_calc_model_stats(predictions[i], answers)
#    for ac, se, sp, name in zip(acc,sens,spec,['Hyperplasic','Serrated','Adenoma']):
#        print(name+ ' stats: ')
#        print('Accuracy: {0:.2f}%'.format(round(ac*100,2)))
#        print('Sensitivity: {0:.2f}%'.format(round(se*100,2)))
#        print('Specificity: {0:.2f}%'.format(round(sp*100,2)))
#        print('===================')
