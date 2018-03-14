# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 02:33:50 2018

@author: Linus
"""

#IMPORTS
#------------------------------------------------------------------------------
import numpy as np
import csv
import time
import os
import argparse
from helpers import *
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import graphviz 
from sklearn import tree
#from helpers import predict_with_file
#from helpers import make_classifiers_predict
#from helpers import calc_model_stats
#------------------------------------------------------------------------------

if __name__ == "__main__":
    
    print('Loading in data...')
    dat_tot = []
    with open('feature_data.csv', newline='') as csvfile:
            tot_data = csv.reader(csvfile, delimiter=',')
            for row in tot_data:
                dat_tot.append([float(r) for r in row[1:]])#we cut off the zeroth column as all it has is the names of the data
            print('Data read in successfully...')
    print('classifying benign vs malignant tumors...')
    for i in range(len(dat_tot)):
        if dat_tot[i][0] != 1.0:
            dat_tot[i][0] = 0.0
    full_list = []
    for i in range(0,len(dat_tot)-1,2):
        print('fold: {}'.format(i))
        y=[]; X= []
        for row in range(0,len(dat_tot)-1,2):
            if row != i:
                y.append(dat_tot[row][0])
                y.append(dat_tot[row+1][0])
                X.append(dat_tot[row][1:])
                X.append(dat_tot[row+1][1:])

        clf = RandomForestClassifier(warm_start=True, oob_score=True, random_state=123)

        '''
        d_tree = clf.estimators_[0]
        dot_data = tree.export_graphviz(d_tree, out_file='tree.dot', 
                             class_names=['malignant', 'benign'],  
                             filled=True, rounded=True)
        graph = graphviz.Source(dot_data) 
        '''

        error_rate = []
        min_estimators= 20; max_estimators = 1100
        for i in range(min_estimators, max_estimators + 1,10):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate.append(oob_error)
        full_list.append(error_rate)
    
    full_list = np.array(full_list)
    
    xs = list(range(20,1101,10))
    ys= []
    for column in full_list.T:
        ys.append(np.average(column))
    plt.plot(xs, ys)
    
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.show()
