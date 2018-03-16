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
import sklearn.metrics as mt

class FlagError(Exception):
    def __init__(self, message):
        self.message = message

def split(row_indcs, folds):
    k, m = divmod(len(row_indcs), folds)
    return (row_indcs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(folds))

def multi_calc_model_stats(predictions, answers):
    '''args: predictions: the list of predictions shape(76,2) for each lesion
             answers: the list of answers for each lesion
       returns: acc: list of accuracy for the three states
                sens: list of sensitivities for the three states
                spec: list of specificities for the three states
                tots: list of overall acc in [0] and f1 score in [1]
    '''
    #Hyp == 1 Ser ==2 Aden ==3
    #1 is true positive, true negative, false positive, false negative 
    confu_matrix = mt.confusion_matrix(answers,predictions, labels = np.array([1,2,3])) 
    total = np.sum(confu_matrix)
    print('Confusion Matrix: ')
    print(confu_matrix)
    total = np.sum(confu_matrix)
    tots = [(confu_matrix[0,0]+confu_matrix[1,1]+confu_matrix[2,2])/total,
            mt.fbeta_score(answers,predictions,labels=[0,1,2],beta=2,average='micro')]
    Hyp = [0,0,0,0]
    Ser = [0,0,0,0]
    Aden = [0,0,0,0]
    Stats = [Hyp,Ser,Aden]
    classes = [0,1,2]
    for predic, ans in zip(predictions, answers):
        p,a = (int(predic),int(ans))
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
    return acc,sens,spec,tots

def binary_calc_model_stats(predictions, answers):
    '''args: predictions: the list of predictions shape(76,2) for each lesion
             answers: the list of answers for each lesion
       returns: acc: accuracy for the two states
                sens: sensitivity for the two states
                spec: specificity for the two states
                f1: f1 score of the classifier
    '''
    #makes a confusion matrix 
    #[0-actual 0 ,  1-actual 0]
    #[0-actual 1 , 1 - actual 1]
    confu_matrix = mt.confusion_matrix(answers,predictions, labels = np.array([0,1]))   
    total = np.sum(confu_matrix)
    print('Confusion Matrix: ')
    print(confu_matrix)
    acc = float(confu_matrix[0,0]+confu_matrix[1,1])/total
    spec = confu_matrix[0,0]/float(confu_matrix[0,0]+confu_matrix[0,1])
    sens = confu_matrix[1,1]/float(confu_matrix[1,0]+confu_matrix[1,1])
    f1 = mt.fbeta_score(answers,predictions, beta=2)
    return acc, sens, spec, f2