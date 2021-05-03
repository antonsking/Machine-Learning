'''
 _____                              _   _   _           _            ___  ___           _     _            
/  ___|                            | | | | | |         | |           |  \/  |          | |   (_)           
\ `--. _   _ _ __  _ __   ___  _ __| |_| | | | ___  ___| |_ ___  _ __| .  . | __ _  ___| |__  _ _ __   ___ 
 `--. \ | | | '_ \| '_ \ / _ \| '__| __| | | |/ _ \/ __| __/ _ \| '__| |\/| |/ _` |/ __| '_ \| | '_ \ / _ \
/\__/ / |_| | |_) | |_) | (_) | |  | |_\ \_/ /  __/ (__| || (_) | |  | |  | | (_| | (__| | | | | | | |  __/
\____/ \__,_| .__/| .__/ \___/|_|   \__|\___/ \___|\___|\__\___/|_|  \_|  |_/\__,_|\___|_| |_|_|_| |_|\___|
            | |   | |                                                                                      
            |_|   |_|                                                                                     
'''

import numpy as np
from random import uniform
from sklearn.preprocessing import MinMaxScaler

class SVM:
    def __init__(self,d,lambda_val=1,max_iter=1000,eta=1e-3):
        self.d = d
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.eta = eta
        
        #weights
        self.w0 = 0
        self.w = np.zeros(self.d)

    def fit(self,X,r):
        self.w0 = 0

        # scaling
        self.MMS = MinMaxScaler().fit(X)
        X = self.MMS.transform(X)
        
        # error function terms
        regular = self.lambda_val / 2
        N = X.shape[0]
        coef = 1 / N
        error = [1]
        
        for iters in range(self.max_iter):
            errsum = 0
            
            for t in range(N):
                
                if r[t] * (X[t] @ self.w - self.w0) < 1: 
                    
                    # error function
                    err = 1 - r[t] * ( X[t] @ self.w - self.w0 )
                    errsum += err
                    
                    # Gradient descent update
                    self.w -= self.eta * (self.lambda_val * self.w - np.dot(X[t] , r[t]) )
                    self.w0 -= self.eta * r[t]
                
            # error result
            izer = np.linalg.norm(self.w,ord=2)**2
            regularizer = regular * izer        # one of the better lines of code ive ever written
            result = np.sum( coef * errsum + regularizer )
            error.append(result)
            
            # convergence check
            if abs(error[iters-1] - error[iters]) < 1e-6:
                return self.w
            
        return self.w
    
    def predict(self,X):
        X = self.MMS.transform(X)
        r_pred = np.sign(X @ self.w)

        return r_pred