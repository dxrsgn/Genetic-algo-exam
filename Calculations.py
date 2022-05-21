import numpy as np
import numpy.linalg as linalg

#Calculating theoretical max W for our problem. For some reason it doesn't work :/
def calculate_W_max(W, B, C):
    inv_B = linalg.inv(B)
    W_max = np.dot(inv_B @ C, C) * (-0.5)
    return W_max

#Calculating naive upper bound of W
def calculate_my_W_max(W, B, C, sigma, h):
    ones = []
    for col in W.T:
        ones.append(np.count_nonzero(col == 1))
    f = [sigma(x - h) for x in ones]
    return np.dot((B@f), f)/2 + np.dot(C,f)

# def just_another_W_max(W, B, C,M):
    # inv_B = linalg.inv(B)
    # f = -(inv_B @ C)
    # return np.dot((B@f), f)/2 + np.dot(C,f)

#Calculate trait function vector f: f_i = sigmoid(sum(W[i][j]*s[i]) - h)    
def calculate_F_vector(genotype, W, h, sigma):
    f_vector = []
    for col in W.T:
        f_vector.append(sigma(sum(col*genotype) - h))
    return np.asarray(f_vector)
#Calculating W for our fitness
def calculate_W(f_vector, B, C):
    return np.dot((B@f_vector), f_vector)/2 + np.dot(C,f_vector)
    
