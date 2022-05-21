import numpy as np
import matplotlib.pyplot as plt
import Generation as gen
import Mutators as mut
import math
from Calculations import *

#Fitness function: coef_C * exp(coef_Lambda * W(s)), where W(s) = (B*f_vector, f_vector)/2 + (C, f_vector)

class SSWM:
    def __init__(self, N, M, K, T_stop = 500, h = 1.5, p_mut = None, \
                coef_Lambda = 1.1, coef_Beta = 0.85, coef_C = 1.1, exp_fitness = False, **kwargs):
                        
        if p_mut == None:
            self.p_mut = 1/N #Mutation probability
        else:
            self.p_mut = p_mut
        
        self.M = M #Number of traits
        self.N = N #Number of genes
        self.K = K #Number of nonzero weights for matrix W
        self.T_stop = T_stop #Max number of generations
        self.h = h #Treshold
        self.coef_Lambda = coef_Lambda
        self.coef_Beta = coef_Beta
        self.coef_C = coef_C
        
        #Way of distributing nonzero weights (-1 or 1) for matrix W
        #"fair" - fair coin, "poisson" - poisson distribution
        self.pleiotropy = kwargs.get('pleiotropy', "fair")
        
        #Rewrite!!!
        #Making copy of model with slightly different parameters. It is used for debug purposes
        if 'W' in kwargs and 'B' in kwargs and 'C' in kwargs and 'genotype' in kwargs:
            self.W = kwargs.get('W')
            self.B = kwargs.get('B')
            self.C = kwargs.get('C')
            self.genotype = kwargs.get('genotype')
            
        #Generating matrices W, B, C
        else:
            self.W, self.B, self.C = gen.generateWBC(N, M, K, self.pleiotropy)
            self.genotype = None
        
        #Sigmoid function for calculting trait function
        #"none" - sigma(x) = x, "logistic" - sigma(x) = 1/(1 + exp(-x))
        sigma_type = kwargs.get('sigma', "none")
        if sigma_type == "logistic":
            self.sigma = lambda x : 1/(1 + math.exp(-x))
        elif sigma_type == "none":
            self.sigma = lambda x : x
        
        #Type of function that changes the genotype.
        mutator_type = kwargs.get('mutator', "flip")
        if mutator_type == "flip":
            self.mutator = mut.flip
            
        #Sometimes overflow occurs due to exponent, in order to prevent this there is exp_fitness variable
        #if exp_fitness == True, fitness calculated with exponent, otherwise exponent omitted
        if exp_fitness:
            self.fitness_type = lambda x : self.coef_C * math.exp(x*coef_Lambda)
        else:
            self.fitness_type = lambda x : x
        
        #F_max - maximal theoretical fitness for given W, B, C
        self.F_max = self.fitness_type(calculate_my_W_max(self.W,self.B,self.C,self.sigma, self.h))
        #self.F_max = self.fitness_type(calculate_W_max(self.W,self.B,self.C))
        #self.F_max = self.fitness_type(just_another_W_max(self.W, self.B, self.C, M))
        
        self.fitness_history = []
        self.generation_history = []
        self.fittest = None
        self.first_genotype = None
    
    def calculate_Fitness(self, f_vector):
        return self.fitness_type(calculate_W(f_vector, self.B, self.C))
    
    def evolve(self):
        #Generating population if it's not given
        if type(self.genotype) == type(None):
            self.genotype = gen.generateGenPool(self.N)
            
        #Save first genotype in order to recreate this model later
        self.first_genotype = self.genotype.copy()
        
        #Calculating fitness
        f_vector = calculate_F_vector(self.genotype, self.W, self.h, self.sigma)
        cur_fitness = self.calculate_Fitness(f_vector)
        
        T = 0
        #Evolving until T reaches T_stop or until we (almost) completely adapt our genotype
        while T < self.T_stop and cur_fitness < self.coef_Beta*self.F_max: 
            f_vector = calculate_F_vector(self.genotype, self.W, self.h, self.sigma)
            cur_fitness = self.calculate_Fitness(f_vector)
            
            #Mutating current genotype and calculating its fitness
            new_genotype = self.mutator(self.genotype.copy(), self.p_mut)
            new_f_vector = calculate_F_vector(new_genotype, self.W, self.h, self.sigma)
            new_fitness = self.calculate_Fitness(new_f_vector)
            
            #Strong selection. If old genotype is less adapted than new one,
            #then replace it with new genotype
            if (new_fitness > cur_fitness):
                self.genotype = new_genotype
            
            #Saving history
            self.fitness_history.append(cur_fitness)
            self.generation_history.append(T)
            self.fittest = self.genotype
            
            T += 1
            
    #Fitness graph        
    def show(self):
        plt.axhline(y=self.F_max, color='r', label='F max')
        plt.plot(self.generation_history, self.fitness_history, 'b', label='Fitness')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()
    
    #Get matrices W, B and vector C, genotype for debug purposes
    def getWBCGenotype(self):
        return (self.W, self.B, self.C, self.first_genotype)
    
    #Return highest fitness found
    def getFmax(self):
        return max(self.fitness_history)
        
    #Return max theoretical fitness 
    def getTheoWmax(self):
        return self.fitness_type(calculate_my_W_max(self.W,self.B,self.C,self.sigma, self.h))
        #return just_another_W_max(self.W,self.B,self.C,self.M)
        #return calculate_W_max(self.W,self.B,self.C)
    