import numpy as np

#Generating matrices W, B, C
def generateWBC(N, M, K, pleiotropy):    
    #Generating W NxM matrix, that holds weights w_i_j for f_j(s) = sigmoid(sum(w_i_j*s_i) - h)
    #Element a[i, j] means that  i-th gene affects j-th trait
    W = np.zeros((N, M))
    
    #Poisson dist for nonzero weights. It consists of Ki - values representing number of non-zero weights
    if pleiotropy == "poisson":
        poisson_distr = np.random.poisson(K, size=(N,))
        for i in range(N):
            #Generating -1 and 1 weights
            weights = np.random.choice([1,-1], size=poisson_distr[i], p=[0.5, 0.5])
            #Generating M - Ki zero weights
            #To fix: If M approximately equal to K, there could occur exception,because poisson_distr[i] may be greater than M 
            weights = np.concatenate((weights, np.zeros((M - poisson_distr[i], ))))
            
            W[i] = np.random.permutation(weights)
        W = W.astype(int)
        
    #Fair coin  
    elif pleiotropy == "fair":
        for i in range(N):
            weights = np.random.choice([1,-1], size = K, p=[0.5, 0.5])
            weights = np.concatenate((weights, np.zeros((M-K, ))))
            W[i] = np.random.permutation(weights)
            
    #Generating B
    #B = np.random.rand(M, M)
    B = np.random.randint(1, 5, size = (M,M))
    B = np.tril(B) + np.tril(B, -1).T
    
    #Generating C
    #C = np.random.rand(M, )
    C = np.random.randint(1, 5, size = (M,))
    
    return (W, B, C)
    
#Generating boolean vector 
def generateGenPool(N):
    return np.random.choice([0,1], (N, ))
