'''
 _   __                                     
| | / /                                     
| |/ /______ _ __ ___   ___  __ _ _ __  ___ 
|    \______| '_ ` _ \ / _ \/ _` | '_ \/ __|
| |\  \     | | | | | |  __/ (_| | | | \__ \
\_| \_/     |_| |_| |_|\___|\__,_|_| |_|___/
                                          
  
'''

import numpy as np
np.random.seed(0)

def kmeans(X,k=5):
    max_iter = 100

    N = len(X) # number of samples
    m_idx = np.random.permutation(N)
    mu = X[m_idx[:k],:] # (k x 3) <-- mu starts with random rows in X
    
    X   = np.delete(X,3,axis=1)
    mu = np.delete(mu,3,axis=1)

    E = []

    for i in range(max_iter):
        r = [ [0 for x in range(k)] for y in range(N) ] # N x k, k labels for each N samples

        # update r
        for t_ in range(N):
            for k_ in range(k):
                r[t_][k_] =  np.linalg.norm(X[t_] - mu[k_])**2 #set r to raw 2 norm squared

            r[t_] = [(r[t_][idx] == min(r[t_])) for idx in range(k)] #set r to booleans representing max of all 2 norms
            
        # Calculate the total reconstruction error define in Textbook Eq.(7.3) 
        iteration_error = 0
        for t_ in range(N):
            for k_ in range(k):
                if(r[t_][k_] == 1):
                    iteration_error += np.linalg.norm(X[t_] - mu[k_])**2 #total error summed up

        E.append(iteration_error) #append total error onto error list

        print('Iteration {}: Error {}'.format(i,E[i]))

        # update mu
        for k_ in range(k):
            rtk = 0
            xrtk = 0
            for t_ in range(N):
                if(r[t_][k_] == 1):
                    rtk += 1 # counter representing number of elements in this class
                    xrtk += X[t_] # counter summing the raw value at X of that element
            
            mu[k_] = xrtk / rtk #taking average of raw value by number of elements

        if (i>1 and (E[i-1] - E[i] < 0.000001)): # if on at least second iteration and error is low enough break loop
            break

    return r,mu,E