'''
___  ___      _ _   _      _               _____                     _             
|  \/  |     | | | (_)    | |             |  __ \                   (_)            
| .  . |_   _| | |_ _  ___| | __ _ ___ ___| |  \/ __ _ _   _ ___ ___ _  __ _ _ __  
| |\/| | | | | | __| |/ __| |/ _` / __/ __| | __ / _` | | | / __/ __| |/ _` | '_ \ 
| |  | | |_| | | |_| | (__| | (_| \__ \__ \ |_\ \ (_| | |_| \__ \__ \ | (_| | | | |
\_|  |_/\__,_|_|\__|_|\___|_|\__,_|___/___/\____/\__,_|\__,_|___/___/_|\__,_|_| |_|
                                                                                   
                                                                                   
'''

import numpy as np
import math

class MulticlassGaussian:
#Constructor
    def __init__(self,k,d,diag):
    #Instance Variables
        self.k = k
        self.d = d
        self.diag = diag

    #Default Variables
        self.prior = [1/k for i in range(k)]#  list of k priors
        self.mean = [(np.array([0 for i in range(d)])) for i in range(k)]#  list of k mean arrays, d * k
        self.cov = [(np.array([np.eye(d) for i in range(d)])) for i in range(k)]#  list of k coviarance matricies, d * k

    def fit(self,X,r):
        n_points = r.shape[0] #data amount information

        for k in range(self.k): #for each class     
            class_X = X.copy()
            class_sum = 0
            rows_removed = 0
            for d in range(n_points): #for each data point
                if(r.item(d) == k): #if data point is in class k
                    class_sum += 1 #increment number of data points in that class
                else:
                    class_X = np.delete(class_X,d-rows_removed,0)#if not in class, remove that row in X
                    rows_removed += 1

            self.prior[k] = class_sum/n_points # divide number of data points in class k by total
            self.mean[k] = np.mean(class_X,axis=0) # insert means based on class data points
            self.cov[k] = np.cov(X,rowvar=False) #covariance matricies
            if (self.diag == True):
                self.cov[k] = np.diag(np.diag(self.cov[k])) #if diag=True, make cov diagonal

    def discriminant(x,prior,mean,cov):
        cov_det = np.linalg.det(cov)
        x_mean = x-mean 
        maha = np.matmul((np.matmul(x_mean , np.linalg.inv(cov))),x_mean) # mahalanobis distance
        g_x = (-0.5 * math.log(cov_det)) - (0.5 * maha) + math.log(prior)
        return g_x

    def predict(self,X):
        output = []
        
        for x in X: # for every data point
            g_k = [0 for i in range(self.k)]
            for k in range(self.k): #test each class
                g_k[k] = MultiGaussClassify.discriminant(x,self.prior[k],self.mean[k],self.cov[k])
            output.append(g_k.index(max(g_k))) # append class label prediction to output
        
        return output