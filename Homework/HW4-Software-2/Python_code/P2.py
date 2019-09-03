import sys
import random
import numpy as np
import scipy as sc
import matplotlib as mlp
import matplotlib.pyplot as plt

from matplotlib import rc
from scipy import special

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

def expPDF(x, lambd = 1):
    '''
    Returns the expodential PDF
    @params:
        x (np.array): Points of interest
        lambd (float): hyper-parameter
    '''
    return lambd*np.exp(-lambd*x)

def expQuantile(n, lambd = 1):
    '''
    Randomly samples from the expodential PDF
    @params:
        n (int): Number of samples desired
        lambd (float): hyper-parameter
    '''
    Y = np.random.uniform(0,1,n)
    return -np.log(1-Y)/lambd

def sinPrior(x, mu = 1):
    '''
    Returns the sin^2 prior for this problem
    @params:
        x (np.array): Points of interest
        mu (float): random hyper-parameter
    '''
    return np.sin(mu*np.pi*x)**2

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #====== Plot True Prior, Likelihood, and Posterior =====
    #Set up subplots
    f, ax = plt.subplots(3, 1, figsize=(10, 7))
    f.suptitle('Homework 4 Problem 2(a)', fontsize=14)

    #Parameters
    x =  np.linspace(0,10,500)
    sin_Prior = sinPrior(x)
    gamma_PDF = gammaPDF(1.5, x)

    l, = ax[0].plot(x, sin_Prior, 'b', label='Sine Prior')
    l1, = ax[1].plot(x, gamma_PDF, 'r', label='Gamma Likelihood')
    l2, = ax[2].plot(x, sin_Prior*gamma_PDF, 'g', label='Posterior')

    for i, ax0 in enumerate(ax):
        ax0.set_xlabel(r'$\alpha$')
        ax0.set_xlim((0,10))
        ax0.set_ylim(ymin=0)

    plt.legend( handles=[l, l1, l2], loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5)
    plt.tight_layout(rect=[0,.05, 1, 0.93])

    #====== Plot Exponential Proposal distribution =====
    #Set up subplots
    f, ax = plt.subplots(2, 1, figsize=(10, 5))
    f.suptitle('Homework 4 Problem 2(b)', fontsize=14)

    #Parameters
    x =  np.linspace(0,10,500)
    mu_i = 1.0/5
    exp_Prior = expPDF(x, mu_i)

    l, = ax[0].plot(x, exp_Prior, 'b', label='Exponential Prior')
    l1, = ax[1].plot(x, sin_Prior*gamma_PDF, 'g', label='Original Posterior')

    for i, ax0 in enumerate(ax):
        ax0.set_xlabel(r'$\alpha$')
        ax0.set_xlim((0,10))
        ax0.set_ylim(ymin=0)

    plt.legend( handles=[l, l1], loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=5)
    plt.tight_layout(rect=[0,.05, 1, 0.93])

    #====== Metropolis-Hastings Algorithm =====
    #Sample from proposal exponential distribution
    A = expQuantile(10000,mu_i).tolist()
    a_n = expQuantile(1,mu_i)
    samples = []
    #Now perform the "random walk" that is the M-H algo
    for i, a0 in enumerate(A):
        #Calc M-H acceptance probability
        m = (gammaPDF(1.5, a0)*sinPrior(a0)*expPDF(a_n, mu_i))/(gammaPDF(1.5, a_n)*sinPrior(a_n)*expPDF(a0, mu_i))
        if(random.random() < min(1, m)):
            samples.append(a0)
            a_n = a0
        else:
            samples.append(a_n)

    #====== Plot histogram =====
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    f.suptitle('Homework 4 Problem 2(c)', fontsize=14)
    
    bins = np.linspace(0, 10, 50)
    weights = np.ones_like(samples)/float(len(samples))
    ax.hist(np.array(samples), bins, normed=1, color='blue', alpha = 0.5, edgecolor = "black", label='M-H Samples')
    #Note: normalize the posterior by numerical integral
    ax.plot(x, (sin_Prior*gamma_PDF)/np.trapz(sin_Prior*gamma_PDF, x), 'g', label='Posterior') 
    ax.set_xlim((0,10))
    ax.set_ylim(ymin=0)
    ax.set_xlabel(r'$\alpha$')
    ax.legend()
    plt.show()