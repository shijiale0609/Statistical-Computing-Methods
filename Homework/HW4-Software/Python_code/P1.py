import sys
import random
import numpy as np
import scipy as sc
import matplotlib as mlp
import matplotlib.pyplot as plt

from matplotlib import rc
from scipy import special

def cauchyPDF(x, x0 , gamma = 1):
    '''
    Returns the cauchy PDF
    @params:
        x (np.array): Points of interest
        x0 (float): Mean location of cauchy
        gamma (float): probably error
    '''
    return 1 / (np.pi*gamma*(1 + (x**2 -2*x*x0 + x0**2)/(gamma**2)))

def cauchy(n, x0 , gamma = 1):
    '''
    Randomly samples from a cauchy PDF
    @params:
        n (int): Number of samples desired
        x0 (float): Mean location of cauchy
        gamma (float): probably error
    '''
    Y = np.random.uniform(0,1,n)
    return gamma*np.tan(np.pi*(Y-0.5)) + x0

def gammaPDF(x, a , b = 1.0):
    '''
    Returns the gamma PDF
    @params:
        x (np.array): Points of interest
        a (float): hyper-parameter
        b (float): hyper-parameter
    '''
    #Note scipy.special.expit = np.exp but np.exp will over flow when x is negative
    #scipy.special.expit dances around that by computing 1/(1+exp(-x))
    return x**(a-1) * sc.special.expit(-x/b - np.log(sc.special.gamma(a)) + np.log(b**a))

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #Set up subplots
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle('Homework 4 Problem 1', fontsize=14)

    #Parameters
    a = 10 # Must be greater than 1
    M = (np.pi*np.sqrt(2*a-1))/sc.special.gamma(a)*np.exp(-a+1)*(a-1)**(a-1)
    x =  np.linspace(0,30,100)

    gamma_PDF = gammaPDF(x, a)
    cauchy_PDF = M*cauchyPDF(x, a-1, np.sqrt(2*a-1))

    #Accept/Reject Monte carlo
    #Get random samples from our "simpler" cauchy distribution
    X = cauchy(5000, a-1, np.sqrt(2*a-1)).tolist()
    #Now accept/reject the sample points
    samples = [ x0 for x0 in X  if (random.random() < gammaPDF(x0, a)/(M*cauchyPDF(x0, a-1, np.sqrt(2*a-1))))]
    bins = np.linspace(0, 30, 30)

    #Plot profiles
    ax[0].plot(x, gamma_PDF, 'g', label=r'$Gamma(x|a,1)$')
    ax[0].plot(x, cauchy_PDF, 'r', label=r'$Cauchy(x|a-1,\sqrt{2a-1})$') 
    ax[0].set_xlim((0,30))
    ax[0].set_ylim(ymin=0)
    ax[0].legend()
    ax[0].set_title('Profiles')
    
    #Plot histrogram
    ax[1].hist(np.array(samples), bins, color='blue', alpha = 0.5, normed=1, edgecolor = "black")
    ax[1].plot(x, gamma_PDF, 'g', label=r'$Gamma(x|a,1)$')
    ax[1].set_xlim((0,30))
    ax[1].set_ylim(ymin=0)
    ax[1].legend()
    ax[1].set_title('Accept/Reject')
    plt.show()