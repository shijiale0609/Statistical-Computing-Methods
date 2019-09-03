'''
Nicholas Geneva
ngeneva@nd.edu
November 3, 2017
'''
import sys
import random
import numpy as np
import scipy as sc
import matplotlib as mlp
import matplotlib.pyplot as plt

from matplotlib import rc
from scipy import special
from scipy import stats

def markovChainWeight(x0, b0 = -1, b1 = 1, tk = 'uniform'):
    '''
    Completes a Markov chain for SIS MC integrals.
    See Ref: https://doi.org/10.1016/j.amc.2010.03.138
    Args:
        x0 (float) = point of interest
        b0 (float) = Lower bound of integral
        b1 (float) = Upper bound of integral
        tk (string) = Transition kernel type (uniform, guassian)
    Returns:
        w (float) = calculated weight 
    '''
    k = 1
    #Construct markov chain
    if(tk == 'uniform'):
        while(True):
            #Transition kernal to next location
            x1 = np.random.uniform(b0-0.5,b1+0.5)
            pd = 1.0 - (b1-b0)/(b1-b0+1.0) #Probability of killing the chain
            p = 1.0/(b1-b0+1) #General probability

            if(x1 > b1 or x1 < b0): #Kill markov chain
                break
            k = k * 0.5*(x1 - x0)/p
            x0 = x1

    elif(tk == 'guassian'):
        mu = 0
        sigma = 0.75
        #Probability of killing the chain
        pd = 1.0 - (sc.stats.norm.cdf(b1, mu, sigma) - sc.stats.norm.cdf(b0, mu, sigma))
        while(True):
            #Transition to 
            x1 = np.random.normal(mu,sigma)
            p =  sc.stats.norm.pdf(x1, mu, sigma)
            if(x1 > b1 or x1 < b0): #Kill markov chain
                break

            k = k * 0.5*(x1 - x0)/p
            x0 = x1
    else:
        print('Transition Kernel not supported')

    return k * x0/pd

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    N = 10000
    y = [0]
    for i in range(N):
        y.append(y[-1] + markovChainWeight(0,b0 = -1,b1 = 1,tk = 'guassian'))
    yNorm = [y0 / float(i+1) for i, y0 in enumerate(y)]

    print('Guassian Estimate: %0.5f' %(y[-1]/float(N)))

    y = [0]
    for i in range(N):
        y.append(y[-1] + markovChainWeight(0,b0 = -1,b1 = 1,tk = 'uniform'))
    yNorm = [y0 / float(i+1) for i, y0 in enumerate(y)]
    
    print('Uniform Estimate: %0.5f' % (y[-1]/float(N)))