'''
  ___      _       _                     _   
 / _ \    | |     | |                   | |  
/ /_\ \ __| | __ _| |__   ___   ___  ___| |_ 
|  _  |/ _` |/ _` | '_ \ / _ \ / _ \/ __| __|
| | | | (_| | (_| | |_) | (_) | (_) \__ \ |_ 
\_| |_/\__,_|\__,_|_.__/ \___/ \___/|___/\__|
                                             
                                             
'''

import numpy as np
from scipy import stats
import random
from math import exp
from sklearn.tree import DecisionTreeClassifier


class Adaboost:
    def __init__(self, num_iters, max_depth=None):
        if num_iters<1:
            raise ValueError

        self.num_iters = num_iters
        self.max_depth = max_depth
        self.trees = []
        self.tree_weights = []

    def fit(self, X, r):
        weights = [ (1/X.shape[0]) for i in range(X.shape[0]) ]
        self.trees = []
        self.tree_weights = []

        for dataset in range(self.num_iters): 
            dtc = DecisionTreeClassifier(criterion='entropy',
                                     random_state=0,
                                     max_depth=self.max_depth)

            # Prepare Dataset
            X_idx = np.array( [ i for i in range(X.shape[0]) ] ) # [ 0 , 1 , 2 , 3 , ... X.shape[0] ]
            Xt_idx = np.random.choice(X_idx, X.shape[0], p=weights)
            Xt = np.take(X,Xt_idx,axis=0)
            rt = np.take(r,Xt_idx,axis=0)

            # Training, error, update weights
            weak = dtc.fit(Xt,rt)
            self.trees.append(weak)
            predicted = dtc.predict(X)
            correct = predicted != r
            error = np.sum(weights * correct) / np.sum(weights)
            alpha = 0.5 * np.log( (1-error) / (error + 1e-10) ) 
            self.tree_weights.append(alpha)
            
            for idx,weight in enumerate(weights):
                weight = weight * exp(-1 * alpha * predicted[idx] * r[idx])

            weights = weights / np.sum(weights)

    def predict(self, X):    
        r_pred = []

        for k in range(self.num_iters):  
            
            # make predictions based on the first k tree classifiers
            class_preds = np.zeros( X.shape[0], dtype=int )
            for clf in range(k+1):
                class_preds = np.vstack( (class_preds , self.tree_weights[clf] * self.trees[clf].predict(X) ) )

            class_preds = np.sign(np.sum(class_preds[1:],axis=0))
            r_pred.append( class_preds )

        return r_pred