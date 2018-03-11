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
    parser.add_argument('--cmode', type = str, default = 'multi', help = 'chooses binary vs multi classification mode')
    parser.add_argument('--folds', type = int, default = -1, help = 'chooses the amount folds for cross validation')
    parser.add_argument('--new_clfs', type = bool, default=False, help ='when True forces production of new classifiers')
    parser.add_argument('--n_trees', type = int, default = 1000, help ='number of trees in the forest')
    FLAGS = parser.parse_args()
    
    print('Loading in data...')
    dat_tot = []
    with open(FLAGS.data_path, newline='') as csvfile:
            tot_data = csv.reader(csvfile, delimiter=',')
            for indx, row in enumerate(tot_data):
                row_in_float = [indx]
                for data_pt in row[1:]:
                    row_in_float.append(float(data_pt))
                dat_tot.append(row_in_float)
            print('Data read in successfully...')
    start_time = time.clock()
        
    #if we want to do binary classification then we change the answers so all benign are 1 and all malignant are 0
    if FLAGS.cmode == 'binary':
        print('classifying benign vs malignant tumors...')
        for i in range(len(dat_tot)):
            if dat_tot[i][1] != 1.0:
                dat_tot[i][1] = 0.0
    else:
        print('classifying with multiple categories')
    #reads in the answers in a row for each row of the data
    answers = np.array([np.array([dat_tot[i][1],dat_tot[i+1][1]]) for i in range(0, len(dat_tot)-1, 2)])
    
    #if we didnt give a fold number then we do LOOCV, change folds to the number of lesions
    if FLAGS.folds == -1:
        folds = len(dat_tot)//2
    else:
        folds = FLAGS.folds
        
    #if the file is already formed then just use it
    if os.path.isfile(FLAGS.cmode + '_' + str(folds) + '_folds_classifiers_trees={}.pkl'.format(FLAGS.n_trees)) and FLAGS.new_clfs!=True:
        predictions = predict_with_file(FLAGS.cmode + '_' + str(folds) + '_folds_classifiers_trees={}.pkl'.format(FLAGS.n_trees), dat_tot, start_time, FLAGS.cmode, folds)
    else:
        predictions = make_classifiers_predict(dat_tot, start_time, folds, FLAGS.cmode, n_trees = 1000)
    
    #we get a bunch of predictions and then we want to convert it into accuracy calcs.
    predictions = np.array(predictions)
    print('Calculating statistics...')
    
    if FLAGS.cmode == 'multi':
        acc, sens, spec = multi_calc_model_stats(predictions,answers)
        for ac, se, sp, name in zip(acc,sens,spec,['Hyperplasic','Serrated','Adenoma']):
            print(name+ ' stats: ')
            print('Accuracy: {0:.2f}%'.format(round(ac*100,2)))
            print('Sensitivity: {0:.2f}%'.format(round(se*100,2)))
            print('Specificity: {0:.2f}%'.format(round(sp*100,2)))
            print('===================')
    else:
        print(binary_calc_model_stats(predictions,answers))
        '''
        acc, sens, spec = binary_calc_model_stats(predictions,answers)
        for ac, se, sp, name in zip(acc,sens,spec,['benign','malignant']):
            print(name+ ' stats: ')
            print('Accuracy: {0:.2f}%'.format(round(ac*100,2)))
            print('Sensitivity: {0:.2f}%'.format(round(se*100,2)))
            print('Specificity: {0:.2f}%'.format(round(sp*100,2)))
            print('===================')
        '''
    