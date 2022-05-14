import numpy as np
import numpy.linalg as linalg

def calculate_W_max(W, B, C):
    inv_B = linalg.inv(B)
    W_max = np.dot(inv_B @ C.T, C) * (-0.5)
    return W_max

#Calculate trait function vector f: f_i = sigmoid(sum(W[i][j]*s[i]) - h)    
def calculate_F_vector(genotype, W, h, sigma):
    f_vector = []
    for row in W.T:
        f_vector.append(sigma(sum(row*genotype) - h))
    return np.asarray(f_vector)

def calculate_W(f_vector, B, C):
    return np.dot((B@f_vector), f_vector)/2 + np.dot(C,f_vector)