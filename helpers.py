# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 11:05:31 2018

@author: willie
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import time
import _pickle as pickle
from random import sample

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
    with open(filename, 'rb') as fid:
            clf_list = pickle.load(fid)
    for i in range(len(clf_list)):
        test_list = clf_list[i][1]
        for lesion in test_list:
            predictions.append(clf_list[i][0].predict([lesion[0][1:],lesion[1][1:]]))
        if i%5 == 0:
            time_passed = (time.clock()-start_time)/60
            print (str(int(i)) + " lesions tested in {:.3} minutes".format(time_passed))
            remaining_time = time_passed/(i+1)*(76-i+1)
            print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    time_passed = (time.clock()-start_time)/60
    print('Done in {:.3} minutes'.format(time_passed))
    return predictions

def split(row_indcs, folds):
    shuffled_row_indcs = row_indcs
    #shuffled_row_indcs = sample(row_indcs, k=len(row_indcs)) 
    k, m = divmod(len(row_indcs), folds)
    return (shuffled_row_indcs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(folds))

def make_classifiers_predict(dat_tot, start_time, folds, mode, n_trees = 1000):
    '''args: dat_tot: the data you are predicting on (for this function it holds the answers)
             start_time: mainly for output updates but the time the function started
             folds: the number of folds during cross validations
             n_trees: the number of trees in the forest, defaults to 1000
             mode: whether multi or binary
   returns: the list of predictions for every row in dat_tot
    '''
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
        for lesion in test_list:
            predictions.append(clf.predict([lesion[0][1:],lesion[1][1:]]))
        time_passed = (time.clock()-start_time)/60
        print(str((fold_number+1)*len(test_list)) + " lesions tested in {:.3} minutes".format(time_passed))
        remaining_time = time_passed/((fold_number+1)*len(test_list))*(76-((1+fold_number)*len(test_list)))
        print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    with open(mode + '_' + str(folds) + '_folds_classifiers_trees={}.pkl'.format(n_trees), 'wb') as fid:
        pickle.dump(clf_list, fid)
    time_passed = (time.clock()-start_time)/60
    print('Done in {:.3} minutes'.format(time_passed))
    return predictions
    

def multi_calc_model_stats(predictions, answers):
    '''args: predictions: the list of predictions shape(76,2) for each lesion
             answers: the list of answers for each lesion
       returns: acc: list of accuracy for the three states
                sens: list of sensitivities for the three states
                spec: list of specificities for the three states
    '''
    #Hyp == 1 Ser ==2 Aden ==3
    #1 is true positive, true negative, false positive, false negative 
    Hyp = [0,0,0,0]
    Ser = [0,0,0,0]
    Aden = [0,0,0,0]
    Stats = [Hyp,Ser,Aden]
    classes = [0,1,2]
    for predic, ans in zip(predictions, answers):
        for i in [0,1]:
            p,a = (int(predic[i]),int(ans[i]))
            if p == a:
                Stats[p-1][0] += 1
                for cl in classes:
                    if cl is not p:
                        Stats[cl-1][1] += 1
            else:
                Stats[p-1][2] += 1
                Stats[a-1][3] += 1
    acc = [(c[0]+c[1])/sum(c) for c in Stats]
    sens = [c[0]/(c[0]+c[3]) for c in Stats]
    spec = [c[1]/(c[1]+c[2]) for c in Stats]
    return acc,sens,spec

def binary_calc_model_stats(predictions, answers):
    '''args: predictions: the list of predictions shape(76,2) for each lesion
             answers: the list of answers for each lesion
       returns: acc: accuracy for the two states
                sens: sensitivity for the two states
                spec: specificity for the two states
    '''
    #makes a confusion matrix 
    #[0-actual 0 ,  1-actual 0]
    #[0-actual 1 , 1 - actual 1]
    confu_matrix = np.zeros((2,2))
    for predic, ans in zip(predictions, answers):
        for i in [0,1]:
            p,a = (int(predic[i]),int(ans[i]))
            confu_matrix[a,p]+=1
    total = np.sum(confu_matrix)
    acc = float(confu_matrix[0,0]+confu_matrix[1,1])/total
    spec = confu_matrix[0,0]/float(confu_matrix[0,0]+confu_matrix[0,1])
    sens = confu_matrix[1,1]/float(confu_matrix[1,0]+confu_matrix[1,1])
    return acc, sens, spec