'''
______                _                ______                  _   
| ___ \              | |               |  ___|                | |  
| |_/ /__ _ _ __   __| | ___  _ __ ___ | |_ ___  _ __ ___  ___| |_ 
|    // _` | '_ \ / _` |/ _ \| '_ ` _ \|  _/ _ \| '__/ _ \/ __| __|
| |\ \ (_| | | | | (_| | (_) | | | | | | || (_) | | |  __/\__ \ |_ 
\_| \_\__,_|_| |_|\__,_|\___/|_| |_| |_\_| \___/|_|  \___||___/\__|
                                                                   
                                                                   
'''

import numpy as np
from scipy import stats
import random
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, num_features, num_trees, max_depth=None):
        if num_features<1 or num_trees<1:
            raise ValueError

        self.num_features = num_features
        self.num_trees = num_trees
        self.max_depth = max_depth


    def fit(self, X, r):
        trees = []
        #number of bootstrapped datasets
        for dataset in range(self.num_trees): 
            dtc = DecisionTreeClassifier(criterion='entropy',
                                     random_state=0,
                                     max_depth=self.max_depth,
                                     max_features=self.num_features)
                                     
            dataset_X = np.zeros( X.shape[1] , dtype=int )
            dataset_r = [0]

            #number of elements in each dataset
            for element in range(X.shape[0]):   
                #random element from X,r
                idx = random.randint(0, X.shape[0]-1)   
                #add random datapoint      
                dataset_X = np.vstack( (dataset_X,X[idx]) )
                #add random datapoint's class   
                dataset_r.append(r[idx])
            trees.append( dtc.fit(dataset_X[1:], dataset_r[1:]) ) 
            
        self.trees = trees
        return trees

    def predict(self,X): 
        r_pred = []

        for k in range(self.num_trees):  
            
            # make predictions based on the first k tree classifiers
            class_preds = np.zeros( X.shape[0], dtype=int )
            for clf in range(k+1):
                class_preds = np.vstack( (class_preds , self.trees[clf].predict( X )) )

            modes,counts = stats.mode(class_preds,axis=0)
            r_pred.append( modes )

        return r_pred

