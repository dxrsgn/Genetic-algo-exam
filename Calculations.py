import numpy as np
import numpy.linalg as linalg

#Calculating theoretical max W for our problem.
def calculate_W_max(W, B, C):
    inv_B = linalg.inv(B)
    W_max = np.dot(-(inv_B @ C), -C)*0.5 + np.dot(C, (-inv_B@C))
    return W_max

#Calculating naive upper bound of W.
#Problem: given in lecture W_max formula ignore dependence on s, so
#calculate_W_max function gives very small value relative to W calculated in algorithm
#Function below gives the highest bound for W with positive matrix W,B,C, h and logistic sigma
#This function calculates x_i number of w_i == 1 for each trait and calculates vector f by substitution x_i into f_i 
#It can be interpreted as fitness with each trait having maximal expression
def calculate_naive_W_max(W, B, C, sigma, h):
    ones = []
    for col in W.T:
        ones.append(np.count_nonzero(col == 1))
    f = [sigma(x - h) for x in ones]
    return np.dot((B@f), f)/2 + np.dot(C,f)


def just_another_W_max(W, B, C,M):
    upper_f = -np.sum(C)/np.sum(B)
    print(upper_f, (upper_f**2), (upper_f**2)*np.sum(B) + np.sum(C)*upper_f)
    return (upper_f**2)*np.sum(B) + np.sum(C)*upper_f
    #return np.dot((B@f_vector), f_vector)/2 + np.dot(C,f_vector)

#Calculating trait function vector f: f_i = sigmoid(sum(W[i][j]*s[i]) - h)    
def calculate_F_vector(genotype, W, h, sigma):
    f_vector = []
    for col in W.T:
        f_vector.append(sigma(sum(col*genotype) - h))
    return np.asarray(f_vector)
    
#Calculating W for our fitness
def calculate_W(f_vector, B, C):
    return np.dot((B@f_vector), f_vector)/2 + np.dot(C,f_vector)
    
