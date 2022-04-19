import numpy as np

#Calculate vector F: f_i = sigmoid(sum(W[i][j]*s[i]) - h)
def calculate_F(W, s, h, M, N):
    F = np.zeros(M, )
    for j in range(M):
        f_j = 0
        for i in range(N):
            f_j += W[i][j]*s[i]
        f_j = f_j - h
        
        #Sigmoid
        F[j] = 1/(1 + np.exp(-0.5*f_j))
    return F
   
#Calculate fitness F = coefC * exp(coefLambda * W_fit)
def calculate_Fitness(B, C, W, s, h, coefC, coefLambda, M, N):
    F = calculate_F(W, s, h, M, N)
    
    #W_fit = (B*F, F)/2  + (C, F) or W_fit = sum(C[i]F[i]) + sum(sum(B[i][j]*F[i]*F[j]))
    W_fit = np.dot((B@F), F)/2 + np.dot(C, F)

    return coefC*np.exp(coefLambda*W_fit)

#Simple flip mutation
def mutate(s, N, N_pop, p_mut):
    for i in range(N_pop):
        for j in range(N):
            r = np.random.sample()
            if (r <= p_mut):
                s[j, i] = 1 - s[j, i]
    return s   