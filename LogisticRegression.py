'''

 _                 _     _   _       ______                             _             
| |               (_)   | | (_)      | ___ \                           (_)            
| |     ___   __ _ _ ___| |_ _  ___  | |_/ /___  __ _ _ __ ___  ___ ___ _  ___  _ __  
| |    / _ \ / _` | / __| __| |/ __| |    // _ \/ _` | '__/ _ \/ __/ __| |/ _ \| '_ \ 
| |___| (_) | (_| | \__ \ |_| | (__  | |\ \  __/ (_| | | |  __/\__ \__ \ | (_) | | | |
\_____/\___/ \__, |_|___/\__|_|\___| \_| \_\___|\__, |_|  \___||___/___/_|\___/|_| |_|
              __/ |                              __/ |                                
             |___/                              |___/                                 

'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LogisticRegression:
    def __init__(self,iterations=1300,eta=.13):
        self.iterations = iterations
        self.eta = eta

    def preprocess(self,X):
        MMS = MinMaxScaler()
        return MMS.fit_transform(X)

    def fit(self, X, r):
        errors = []

        # Preprocess input
        X = self.preprocess(X)
        
        # Set weights to 0
        self.w = [0 for i in range(X.shape[1])]
        
        for iters in range(self.iterations):
            
            # w^T * x
            wTx = X @ self.w
            
            # P(C_1 | x)
            posteriors = 1 / (1 + np.exp(wTx))
            
            # gradient descent
            df = posteriors - r
            grad = (X.T @ df) 
            self.w += self.eta * grad
            
            # Check for convergence
            error = -1 * (r * np.log(posteriors) + (1-r) * np.log(1-posteriors)).mean()
            errors.append(error)
            if iters > 1:
                if errors[iters-1] - errors[iters] < 1e-6:
                    return self.w
                
        return self.w
    
    
    def predict(self,X):
        # Preprocess input
        X = self.preprocess(X)
        
        # w^T * x
        wTx = X @ self.w
        
        # resultant sigmoid
        sigmoid = 1 / (1 + np.exp(wTx))

        return sigmoid >= 0.5