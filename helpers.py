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

def predict_with_file(filename, dat_tot, start_time):
    '''args: filename: string with the file holding your classifiers
            dat_tot: the data you are predicting on (for this function it holds the answers)
            start_time: mainly for output updates but the time the function started
       returns: the list of predictions for every row in dat_tot
    '''
    predictions = []
    clf_list = []
    with open(filename, 'rb') as fid:
            clf_list = pickle.load(fid)
    for i in range(len(clf_list)):
        predictions.append(clf_list[i].predict([dat_tot[2*i][2:],dat_tot[2*i+1][2:]]))
        if i%5 == 0:
            time_passed = (time.clock()-start_time)/60
            print (str(int(i)) + " lesions tested in {:.3} minutes".format(time_passed))
            remaining_time = time_passed/(i+1)*(76-i+1)
            print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    print('Done!')
    return predictions

def make_classifiers_predict(dat_tot, start_time, n_trees = 1000):
    '''args: dat_tot: the data you are predicting on (for this function it holds the answers)
             start_time: mainly for output updates but the time the function started
             n_trees: the number of trees in the forest, defaults to 1000
   returns: the list of predictions for every row in dat_tot
    '''
    predictions = []
    clf_list = []
    for i in range(0, len(dat_tot)-1, 2):
         dat_tot2 = dat_tot[0:i+1] + dat_tot[i+2:]
         y = []; X = []
         for sample in dat_tot2:
             y.append(sample[1])
             X.append(sample[2:])
         clf = RandomForestClassifier(n_estimators = n_trees)
         clf.fit(X,y)
         clf_list.append(clf)
         predictions.append(clf.predict([dat_tot[i][2:],dat_tot[i+1][2:]]))
         time_passed = (time.clock()-start_time)/60
         print (str(int(i/2)) + " lesions tested in {:.3} minutes".format(time_passed))
         remaining_time = time_passed/((i+1)/2)*(76-((i+1)/2))
         print ('Approximately {:.2} minutes remaining'.format(remaining_time))
    with open('classifiers_trees={}.pkl'.format(n_trees), 'wb') as fid:
        pickle.dump(clf_list, fid)
    print('Done!')
    return predictions

def calc_model_stats(predictions, answers):
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
# =============================================================================
#             pred = int(predic[i])
#             a = int(ans[i])
#             if pred == 1:
#                 if a == 1:
#                     Hyp[0] += 1
#                     Ser[1]+=1
#                     Aden[1]+=1
#                 if a == 2:
#                     Hyp[2]+=1
#                     Ser[3]+=1
#                 if a == 3:
#                     Hyp[2]+=1
#                     Aden[3]+=1
#             if pred == 2:
#                 if a == 1:
#                     Ser[2]+=1
#                     Hyp[3]+=1
#                 if a == 2:
#                     Ser[0]+=1
#                     Hyp[1]+=1
#                     Aden[1]+=1
#                 if a == 3:
#                     Ser[2]+=1
#                     Aden[3]+=1
#             if pred == 3:
#                 if a == 1:
#                     Aden[2]+=1
#                     Hyp[3]+=1
#                 if a == 2:
#                     Aden[2]+=1
#                     Ser[3]+=1
#                 if a == 3:
#                     Aden[0]+=1
#                     Ser[1]+=1
#                     Hyp[1]+=1
#     acc = [(c[0]+c[1])/sum(c) for c in [Hyp,Ser,Aden]]
#     sens = [c[0]/(c[0]+c[3]) for c in [Hyp,Ser,Aden]]
#     spec = [c[1]/(c[1]+c[2]) for c in [Hyp,Ser,Aden]]
# =============================================================================
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

