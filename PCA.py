'''
______     _            _       _        _____                                              _      ___              _           _     
| ___ \   (_)          (_)     | |      /  __ \                                            | |    / _ \            | |         (_)    
| |_/ / __ _ _ __   ___ _ _ __ | | ___  | /  \/ ___  _ __ ___  _ __   ___  _ __   ___ _ __ | |_  / /_\ \_ __   __ _| |_   _ ___ _ ___ 
|  __/ '__| | '_ \ / __| | '_ \| |/ _ \ | |    / _ \| '_ ` _ \| '_ \ / _ \| '_ \ / _ \ '_ \| __| |  _  | '_ \ / _` | | | | / __| / __|
| |  | |  | | | | | (__| | |_) | |  __/ | \__/\ (_) | | | | | | |_) | (_) | | | |  __/ | | | |_  | | | | | | | (_| | | |_| \__ \ \__ \
\_|  |_|  |_|_| |_|\___|_| .__/|_|\___|  \____/\___/|_| |_| |_| .__/ \___/|_| |_|\___|_| |_|\__| \_| |_/_| |_|\__,_|_|\__, |___/_|___/
                         | |                                  | |                                                      __/ |          
                         |_|                                  |_|                                                     |___/          
'''

import numpy as np 

def PCA(X,k):
# Dataset information
    mean = np.mean(X,axis=0)
    X_norm = X - mean # Normalised
    cov = np.cov(X_norm,rowvar=False)
    vals = np.linalg.eigh(cov)[0]#Eigenvalues
    vecs = np.linalg.eigh(cov)[1]#Eigenvectors

# Select Wanted Components
    components = []
    vals = vals.tolist()
    for i in range(k):
        argmax = vals.index(max(vals)) # Index of largest eigenvalue
        components.append(vecs[:,argmax]) # Add Max Eigenvector to components
        del vals[argmax] # Remove eigenvalue for next iter

    W = np.asarray(components)
    return W, mean

def ProjectDatapoints(X,W,u):
    X_norm = X - u #normalise x
    projection = np.dot(W,X_norm.T).T #project 

    return projection