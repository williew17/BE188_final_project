# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:06:11 2018

@author: linus
"""

import sklearn.cross_decomposition as skcd
import numpy as np
import csv
import argparse
from random import shuffle
import matplotlib.pyplot as plt

from helpers import multi_calc_model_stats, binary_calc_model_stats

#move to helpers.py later
class FlagError(Exception):
    def __init__(self, message):
        self.message = message

def PLS_DA(cmode, fold_quantity, components):

    print(
    '''-------------------------------------------------------------------------------
    Gastrointestinal Lesion Classifier (by Willie Wu and Linus Chen): PLS-DA
    -------------------------------------------------------------------------------''')
    
    #load data
    feature_data=[]
    with open('feature_data.csv', newline='') as csvfile:
            data_reader = csv.reader(csvfile, delimiter=',')
            for row in data_reader:
                feature_data.append([int(row[1])] + [float(r) for r in row[2:]])
            print('Data read in successfully...')
    
    #check input flags
    if cmode not in ['binary', 'multi', 'debug']:
        raise FlagError('Only binary and multi modes are available.')
    if fold_quantity > len(feature_data)//2 or fold_quantity < -1:
        raise FlagError('Folds must be >=-1 (LOOCV) and <= length of data.')
    if components > len(feature_data)//2 or components < 1 :
        raise FlagError('Components must be > 0 and <= length of data.')
    
    #benign and malignant classes
    classes = ['Hyp (b)', 'Ser (m)', 'Ade (m)']
    #number of PC in models
    model = skcd.PLSRegression(components)
    
    #pair indicates that the 2 rows were combined into one
    pair_data = np.array([feature_data[i][1:] + feature_data[i+1][1:] for i in range(0, len(feature_data)-1, 2)])
    pair_answers = np.array([feature_data[i][0] for i in range(0, len(feature_data)-1, 2)])
    pair_modified_answers = []
    
    #multiclass prediction is 3 runs of PLSR-DA, for x vs. not-x
    #for binary, only check the first run of hyp vs. not-hyp
    #3 different sets of answers for each type of classification
    for i in range(len(classes)):
        pair_modified_answers.append([1 if pair_answers[j] == (i+1) else -1 for j in range(0,len(pair_answers))])
    pair_modified_answers = np.array(pair_modified_answers)
    
    if fold_quantity == -1:
        fold_quantity = len(pair_data)
    
    #randomly choose folds for each lesion, approx. equal sizes
    folds = [[] for a in range(fold_quantity)]
    fold_membership = [i%fold_quantity for i in range(len(pair_data))]
    shuffle(fold_membership)
    for b in range(len(fold_membership)):
            folds[fold_membership[b]].append(b)
    
    
    #make predictions for each fold and each class
    test_pair_predictions = []
    test_pair_answers = []
    for f in range(fold_quantity):
        
        fold_test_pair_predictions = []
        test_fold = folds[f]
        train_fold = [i for i in range(len(pair_data)) if i not in test_fold]
        
        for c in range(len(classes)):
            
            #set training data
            train_pair_data = pair_data[train_fold]
            train_pair_modified_answers = pair_modified_answers[c][train_fold]
            
            #set test data
            test_pair_data = pair_data[test_fold]
            
            #fit model to training data, predict
            model.fit(train_pair_data, train_pair_modified_answers)
            fold_test_pair_predictions.append(model.predict(test_pair_data).flatten())
    
        test_pair_predictions.append(np.swapaxes(fold_test_pair_predictions,0,1))
        test_pair_answers.append(pair_answers[test_fold])
    
    #find which class has the highest predicted value
    if cmode in ['binary', 'debug']:
        predictions = [1 if ff[0] > 0 else 0 for f in test_pair_predictions for ff in f]
        answers = [1 if aa == 1 else 0 for a in test_pair_answers for aa in a]
        acc, sens, spec, f1 = binary_calc_model_stats(predictions, answers)
        if cmode != 'debug':
            print('F1 Score: {0:.2f}%'.format(round(f1*100,2)))
            print('Accuracy: {0:.2f}%'.format(round(acc*100,2)))
            print('Sensitivity: {0:.2f}%'.format(round(sens*100,2)))
            print('Specificity: {0:.2f}%'.format(round(spec*100,2)))
            print('===================')
        else:
            return acc, sens, spec, f1, np.array(model.x_loadings_), np.array(model.y_loadings_)
    
    if cmode == 'multi':
        predictions = [np.argmax(f) + 1 for f in test_pair_predictions for ff in f]
        acc, sens, spec, tots = multi_calc_model_stats(predictions, pair_answers)
        print('Total Accuracy: {0:.2f}%'.format(round(tots[0]*100)))
        print('F1 Score: {0:.2f}%'.format(round(tots[1]*100,2)))
        for ac, se, sp, name in zip(acc,sens,spec,['Hyperplasic','Serrated','Adenoma']):
            print(name+ ' stats: ')
            print('Accuracy: {0:.2f}%'.format(round(ac*100,2)))
            print('Sensitivity: {0:.2f}%'.format(round(se*100,2)))
            print('Specificity: {0:.2f}%'.format(round(sp*100,2)))
            print('===================')

# =============================================================================
# if __name__ == '__main__':    
#     #cmd line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cmode', type = str, default = 'binary', help = 'chooses binary vs multi classification mode')
#     parser.add_argument('--folds', type = int, default = -1, help = 'chooses the amount folds for cross validation. Default is LOOCV')
#     parser.add_argument('--components', type = int, default = 30, help = 'number of principal components in the model')
#     FLAGS = parser.parse_args()
#     PLS_DA(FLAGS.cmode, FLAGS.folds, FLAGS.components)
# =============================================================================

#grab figures
if __name__ == '__main__':    
     #cmd line arguments
     parser = argparse.ArgumentParser()
     parser.add_argument('--cmode', type = str, default = 'binary', help = 'chooses binary vs multi classification mode')
     parser.add_argument('--folds', type = int, default = -1, help = 'chooses the amount folds for cross validation. Default is LOOCV')
     parser.add_argument('--components', type = int, default = 27, help = 'number of principal components in the model')
     FLAGS = parser.parse_args()
     
     #loadings plot
     output = PLS_DA(FLAGS.cmode, FLAGS.folds, FLAGS.components)
     fig = plt.figure(figsize=(10,8)) 
     plt.scatter(output[4][:,0], output[4][:,1], color='#7890CD')
     plt.scatter(output[5][:,0], output[5][:,1], color='#F06292')
     plt.axhline()
     plt.axvline()
     plt.xlabel('PC 1')
     plt.ylabel('PC 2')
     fig.savefig("Figure2.png")
     
 #=============================================================================
     #plot of components versus F1 score
     dd=[]
     for i in range(1,76):
         output = PLS_DA(FLAGS.cmode, FLAGS.folds, i)
         dd.append(output[3])
     fig = plt.figure(figsize=(10,8)) 
     plt.plot(dd, color='#7890CD')
     plt.xlabel('Components')
     plt.ylabel('F1 Score')
     fig.savefig("Figure1.png")
# =============================================================================