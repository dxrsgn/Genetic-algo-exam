import numpy as np

#Functions that change genotype
    
#Simple gene flipping with p_mut probability
def flip(genotype, p_mut):
    for i in range(len(genotype)):
        if(np.random.sample() <= p_mut):
            genotype[i] = 1 - genotype[i]
    return genotype
    
#more to be implemented....