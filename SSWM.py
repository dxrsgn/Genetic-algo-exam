import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from Generation import generateGenPool
from Calculations import calculate_Fitness


def SSWM(M, N, K, coefLambda, coefC, coefBeta, h, p_mut, T_stop, N_pop, W, B, C):
    C = C.reshape(-1,)
    fitnessValues = []
    
    #Calculating max W and Fitness
    W_max = np.dot(((linalg.inv(B))@C), C)
    W_max = -W_max/2
    F_max = coefC*np.exp(W_max*coefLambda)
    
    #Generating genetic pool
    s = generateGenPool(N, N_pop)
    F = calculate_Fitness(B, C, W, s, h, coefC, coefLambda, M, N)
    
    print("------DEBUG-----\nInitial F -", F)
    T = 0
    while (T < T_stop and F < coefBeta*F_max):
        F = calculate_Fitness(B, C, W, s, h, coefC, coefLambda, M, N)
        fitnessValues.append(F)
        
        s_mut = mutate(s, N, N_pop, p_mut)
        F_mut = calculate_Fitness(B, C, W, s_mut, h, coefC, coefLambda, M, N)
        if (F_mut > F):
            s = s_mut
        T += 1
    
    plt.plot(range(T), fitnessValues)
    plt.show()
    return (F_max, W_max, fitnessValues)