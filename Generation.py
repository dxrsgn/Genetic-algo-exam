import numpy as np

#Generating matrices W, B, C
def generateWBC(N, M, K):
    #Generating W MxN matrix, that holds weights w_i_j for f_i(s) = sigmoid(sum(w_i_j*s_j) - h_i)
    #Poisson dist for non-zero weights. It consists of Ki - values representing number of non-zero weights
    poisson_distr = np.random.poisson(K, size=(N,))
    
    print("--------------DEBUG---------------")
    print("Average non-zero:", np.absolute(poisson_distr.mean()))
    print("Weights M vector for M traits: ", poisson_distr)
    
    W = np.zeros((N,M))
    for i in range(N):
        #Generating -1 and 1 weights
        weights = np.random.choice([1,-1], size=poisson_distr[i], p=[0.5, 0.5])
        #Generating M - Ki zero weights
        weights = np.concatenate( (weights, np.zeros( (M - poisson_distr[i], ) ) ) )
        
        weights = np.random.permutation(weights)
        W[i] = weights
    W = W.astype(int)

    #Generating B
    B = np.random.uniform(low=1.0, high=10.0, size=(M,M))
    B = np.tril(B) + np.tril(B, -1).T
    
    #Generating C
    C = np.random.uniform(low=1.0, high=10.0, size=(M,))
    
    return (W, B, C)
    
#Generating boolean vector
def generateGenPool(N, N_pop):
    s = np.random.choice([0,1], (N, N_pop))
    return s
    
