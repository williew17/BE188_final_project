# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 19:06:11 2018

@author: willie
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
#from helpers import predict_with_file
#from helpers import make_classifiers_predict
#from helpers import calc_model_stats
#------------------------------------------------------------------------------



if __name__ == "__main__":
    print(
'''-------------------------------------------------------------------------------
Gastrointestinal Lesion Classifier (by Willie Wu and Linus Chen): Random Forest
-------------------------------------------------------------------------------''')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default='feature_data.csv', help = 'directory holding the data')
    parser.add_argument('--cmode', type = str, default = 'binary', help = 'chooses binary vs multi classification mode')
    parser.add_argument('--folds', type = int, default = -1, help = 'chooses the amount folds for cross validation')
    parser.add_argument('--new_clfs', type = bool, default=False, help ='when True forces production of new classifiers')
    parser.add_argument('--n_trees', type = int, default = 1000, help ='number of trees in the forest')
    FLAGS = parser.parse_args()
    
    print('Loading in data...')
    dat_tot = []
    with open(FLAGS.data_path, newline='') as csvfile:
            tot_data = csv.reader(csvfile, delimiter=',')
            for row in tot_data:
                dat_tot.append([float(r) for r in row[1:]])#we cut off the zeroth column as all it has is the names of the data
            print('Data read in successfully...')
    start_time = time.clock()
        
    #if we want to do binary classification then we change the answers so all benign are 1 and all malignant are 0
    if FLAGS.cmode == 'binary':
        print('classifying benign vs malignant tumors...')
        for i in range(len(dat_tot)):
            if dat_tot[i][0] != 1.0:
                dat_tot[i][0] = 0.0
    else:
        print('classifying with multiple categories')
    
    #if we didnt give a fold number then we do LOOCV, change folds to the number of lesions
    if FLAGS.folds == -1:
        folds = len(dat_tot)//2
    else:
        folds = FLAGS.folds
        
    #if the file is already formed then just use it
    if os.path.isfile(FLAGS.cmode + '_' + str(folds) + '_folds_classifiers_trees={}.pkl'.format(FLAGS.n_trees)) and FLAGS.new_clfs!=True:
        predictions, answers = predict_with_file(FLAGS.cmode + '_' + str(folds) + '_folds_classifiers_trees={}.pkl'.format(FLAGS.n_trees), dat_tot, start_time, FLAGS.cmode, folds)
    else:
        predictions, answers = make_classifiers_predict(dat_tot, start_time, folds, FLAGS.cmode, n_trees = FLAGS.n_trees)
    
    #we get a bunch of predictions and then we want to convert it into accuracy calcs.
    classifications = np.array([a[0] for a in predictions])
    scores = np.array([b[1] if b[0] == 0 else 1-b[1] for b in predictions ])
    print('Calculating statistics...')
    
    if FLAGS.cmode == 'multi':
        acc, sens, spec, tots = multi_calc_model_stats(classifications,answers)
        for ac, se, sp, name in zip(acc,sens,spec,['Hyperplasic','Serrated','Adenoma']):
            print(name+ ' stats: ')
            print('Accuracy: {0:.2f}%'.format(round(ac*100,2)))
            print('Sensitivity: {0:.2f}%'.format(round(se*100,2)))
            print('Specificity: {0:.2f}%'.format(round(sp*100,2)))
            print('===================')
        print('Overall accuracy: {0:.2f}%'.format(round(tots[0]*100,2)))
        print('F1-Score: {0:.2f}'.format(round(tots[1]*100,2)))
        print('========================')
    else:
        acc, sens, spec, f1 = binary_calc_model_stats(classifications,answers)
        print('Binary stats: ')
        print('Accuracy: {0:.2f}%'.format(round(acc*100,2)))
        print('Sensitivity: {0:.2f}%'.format(round(sens*100,2)))
        print('Specificity: {0:.2f}%'.format(round(spec*100,2)))
        print('F1-Score: {0:.2f}'.format(round(f1,2)))
        print('===================')
        
        fpr, tpr, x = mt.roc_curve(answers,scores,pos_label=0)
        roc_auc = mt.auc(fpr,tpr)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_random_forest.png')
        
    