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
from sklearn.ensemble import RandomForestClassifier
from random import sample
import _pickle as pickle
#from helpers import predict_with_file
#from helpers import make_classifiers_predict
#from helpers import calc_model_stats
#------------------------------------------------------------------------------

def make_classifiers_predict(dat_tot, start_time, folds, mode, n_trees = 1000):
    '''args: dat_tot: the data you are predicting on (for this function it holds the answers)
             start_time: mainly for output updates but the time the function started
             folds: the number of folds during cross validations
             n_trees: the number of trees in the forest, defaults to 1000
             mode: whether multi or binary
   returns: creates file returns nothing
    '''    
    print('Scrambling data for randomized folds')
    doubled_data = []
    for ii in range(0,len(dat_tot)-1,2):
        doubled_data.append((dat_tot[ii],dat_tot[ii+1]))
    doubled_data = sample(doubled_data, k=len(doubled_data))
    dat_scrambled = []
    for tup in doubled_data:
        dat_scrambled.append(tup[0])
        dat_scrambled.append(tup[1])
    dat_tot = dat_scrambled
    
    #reads in the answers in a row for each row of the data
    answers = np.array([dat_tot[i][0] for i in range(0, len(dat_tot)-1,2)])
    predictions = []
    clf_list = []
    k_fold_list = split(list(range(0,len(dat_tot)-1,2)),folds)
    for fold_number, kfold in enumerate(k_fold_list):
        test_list = []
        y=[]; X= []
        for row in range(0,len(dat_tot)-1,2):
            if row in kfold:
                test_list.append((dat_tot[row],dat_tot[row+1]))
            else:
                y.append(dat_tot[row][0])
                y.append(dat_tot[row+1][0])
                X.append(dat_tot[row][1:])
                X.append(dat_tot[row+1][1:])
        clf = RandomForestClassifier(n_estimators = n_trees)
        clf.fit(X,y)
        clf_list.append((clf,test_list))
        
        #time tracking code
        time_passed = (time.clock()-start_time)/60
        print(str(fold_number+1) + " folds made in {:.3} minutes".format(time_passed))
        remaining_time = time_passed/(fold_number+1)*(folds-1+fold_number)
        print ('Approximately {:.2} minutes remaining'.format(remaining_time))
        
    with open(mode + '_' + str(folds) + '_folds_classifiers_trees={}.pkl'.format(n_trees), 'wb') as fid:
        pickle.dump((clf_list,answers), fid)
        
    time_passed = (time.clock()-start_time)/60
    print('Done in {:.3} minutes'.format(time_passed))

def predict_with_file(filename, dat_tot, start_time, mode, folds):
    '''args: filename: string with the file holding your classifiers
            dat_tot: the data you are predicting on (for this function it holds the answers)
            start_time: mainly for output updates but the time the function started
            folds: the number of folds during cross validation
            mode: whether multi or binary
       returns: the list of predictions for every row in dat_tot
    '''
    predictions = []
    clf_list = []
    answers = []
    with open(filename, 'rb') as fid: #open the pkl file with our classifiers and their matching test lesions
            clf_list, answers = pickle.load(fid)
            
    for i in range(len(clf_list)):#this is essentially every fold
        test_list = clf_list[i][1] #this is what each classifer will be tested against
        for lesion in test_list: #we go through every lesion in the test_list
            pred = clf_list[i][0].predict_proba([lesion[0][1:],lesion[1][1:]]) #we get the probability for each one
            
            #this segment finds the concatenated probability for each lesion to get a single classification for each lesion
            max_col = 0; max_i = -1
            for col in range(pred.shape[1]):
                if sum(pred[:,col]) > max_col:
                    max_col = sum(pred[:,col])
                    max_i = col
            predictions.append((clf_list[i][0].classes_[max_i], max_col/2.))
            
    #time tracking code below    
        if i%5 == 0:
            time_passed = (time.clock()-start_time)/60
            print (str(int(i*len(test_list))) + " lesions tested in {:.3} minutes".format(time_passed))
            remaining_time = time_passed/(i*len(test_list)+1)*(76-i*len(test_list)+1)
            print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    time_passed = (time.clock()-start_time)/60
    print('Done in {:.3} minutes'.format(time_passed))
    
    return predictions, answers

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
                dat_tot.append([int(row[1])] + [float(r) for r in row[2:]])#we cut off the zeroth column as all it has is the names of the data
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
    
    pkl_file_name = FLAGS.cmode + '_' + str(folds) + '_folds_classifiers_trees={}.pkl'.format(FLAGS.n_trees)
    #if the file doesnt exist then make it
    if os.path.isfile(pkl_file_name)==False or FLAGS.new_clfs:
        make_classifiers_predict(dat_tot, start_time, folds, FLAGS.cmode, n_trees = FLAGS.n_trees)
    #use the file
    predictions, answers = predict_with_file(pkl_file_name,dat_tot, start_time, FLAGS.cmode, folds)
    
    #we get a bunch of predictions and then we want to convert it into accuracy calcs
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
        print('F2-Score: {0:.2f}'.format(round(tots[1]*100,2)))
        print('========================')
    else:
        acc, sens, spec, f1 = binary_calc_model_stats(classifications,answers)
        print('Binary stats: ')
        print('Accuracy: {0:.2f}%'.format(round(acc*100,2)))
        print('Sensitivity: {0:.2f}%'.format(round(sens*100,2)))
        print('Specificity: {0:.2f}%'.format(round(spec*100,2)))
        print('F2-Score: {0:.2f}'.format(round(f1,2)))
        print('===================')
        
        fpr, tpr, x = mt.roc_curve(answers,scores,pos_label=0)
        roc_auc = mt.auc(fpr,tpr)
        lw = 2
        fig = plt.figure(figsize=(10,8))
        plt.plot(fpr, tpr, color='#D23369',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='0.3', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_random_forest.png', dpi=400)
        
    