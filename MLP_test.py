# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 02:33:50 2018

@author: Linus
"""

import numpy as np
from sklearn.neural_network import MLPClassifier as MLP
#import time
#import _pickle as pickle

def make_neural(hiddenlayersizes, X, Y, X2):
    neural = MLPClassifier(hidden_layer_sizes=(hiddenlayersizes))
    neural.fit(X, Y)
    return neural.predict(X2)