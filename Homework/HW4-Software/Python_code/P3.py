import sys
import random
import numpy as np
import scipy as sc
import matplotlib as mlp
import matplotlib.pyplot as plt

from matplotlib import rc
from scipy import special

def gaussian1D(x, mu, covar):
    """Calculates 1D gaussian density

        Args:
        x (flost) = point of interest
        mu (float) = mean 
        var (float) = Variance squared 
    """
    e0 = (x-mu)*(covar)
    e0 = np.multiply(e0, (x-mu))
    e = np.exp(-0.5*e0)
    

    return 1.0/(np.power(2.0*np.pi*covar,0.5))*e

def gaussianMD(x, mu, covar):
    """Calculates multi-dimension gaussian density
    Ref: PRML (Bishop) pg. 25
    Args:
        x (np.array) = 1xD array of the point of interest
        mu (np.array) = 1xD array of means 
        covar (np.array) = DxD array matrix 
    """
    D = x.shape[0]
    e = -0.5*(x-mu).T.dot(np.linalg.inv(covar)).dot(x-mu)
    det = np.linalg.det(covar)
    return 1.0/np.power(2*np.pi,D/2.0) * 1.0/(det**0.5) * np.exp(e)

def conditionalNormal(x, mu, covar, givenb=True):
    """Calculates the conditional multi-dimension gaussian density
    Ref: PRML (Bishop) pg. 87
    Args:
        x (np.array) = 1xD array of the point of interest
        mu (np.array) = 1xD array of means 
        covar (np.array) = DxD array matrix 
    """
    D = x.shape[0]
    d = int(D/2)

    mu_a = mu[0:d]
    mu_b = mu[d:]

    covar_aa = covar[0:d, 0:d]
    covar_ab = covar[0:d, d:]
    covar_ba = covar[d:, 0:d]
    covar_bbi = np.linalg.inv(covar[d:, d:])

    if(givenb == True): #Compute p(a|b)
        mu_agb = mu_a - covar_ab.dot(covar_bbi).dot(x[d:]-mu_b) #mu(a|b)
        covar_agb = covar_aa - covar_ab.dot(covar_bbi).dot(covar_ba) #covar(a|b)
        return np.random.multivariate_normal(mu_agb, covar_agb)
    else: #Compute p(b|a)
        mu_bga = mu_b - covar_ba.dot(covar_bbi).dot(x[0:d]-mu_a) 
        covar_bga = covar_bbi + covar_bbi.dot(covar_ba).dot(np.linalg.inv(covar_aa - \
                covar_ab.dot(covar_bbi).dot(covar_ba))).dot(covar_ab).dot(covar_bbi)
        covar_bga = np.linalg.inv(covar_bga)

        return np.random.multivariate_normal(mu_bga, covar_bga)

def leapFrog (x0, p0, nsteps, eps, mu, covar):
    '''
    Performance leapfrog algorithm for nsteps
    Args:
        x0 (nd.array) = 1xD x points
        p0 (nd.array) = 1xD momentum points
        nsteps (int) = number of full steps to take
        eps (float) = step size
        mu (nd.array) = 1xD means of posterior
        covar (nd.array) = DxD covariance of posterior
    '''
    covar_i = np.linalg.inv(covar)
    for i in range(nsteps):
        x_p = x0
        #First half step the momentum
        p_h = p0 - (eps / 2.0)*(x0 - mu).T.dot(covar_i)
        #Full step position
        x0 = x_p + eps*p_h
        #Second half step the momentum
        p0 = p_h - (eps / 2.0)*((x0 - mu).T.dot(covar_i))

    return x0, p0

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #====== Gibbs Sampling =====
    #Set up subplots
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle('Homework 4 Problem 3(a)', fontsize=14)

    #Guassian parameters
    x = np.array([0.5, 0.5])
    mu = np.array([1, 1])
    covar = np.array([[1, -0.5],[-0.5, 1]])

    num_samp = 1000
    g_samp = np.zeros((num_samp, 2))
    #Gibbs sampling loop
    for i in range(2,num_samp):
        g_samp[i,:] =  g_samp[i-1,:]
        #Sample first component
        g_samp[i,0] = conditionalNormal(g_samp[i,:], mu, covar)
        #Sample second component
        g_samp[i,1] = conditionalNormal(g_samp[i,:], mu, covar, False)

    #Plot p(a) histogram/profile
    bins = np.linspace(-4, 5, 100)
    x0 = np.linspace(-4,5,100)
    weights = np.ones_like(g_samp)/100
    ax[0].hist(g_samp[:,0], bins, normed=1, color='blue', alpha = 0.5, edgecolor = "black", label='Gibbs Samples')
    ax[0].plot(x0, gaussian1D(x0.T,mu[0],covar[0,0]), 'r', label=r'$p(a)$')
    #Plot p(b) histogram/profile 
    ax[1].hist(g_samp[:,1], bins, normed=1, color='green', alpha = 0.5, edgecolor = "black", label='Gibbs Samples')
    ax[1].plot(x0, gaussian1D(x0.T,mu[1],covar[1,1]), 'r', label=r'$p(b)$')

    for i, ax0 in enumerate(ax):
        ax0.set_xlabel('x')
        ax0.set_xlim((-4,5))
        ax0.set_ylim(ymin=0)
        ax0.legend()

    #====== Component wise M-H Sampling =====
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 4 Problem 3(b)', fontsize=14)

    #Guassian parameters
    x = np.array([0.5, 0.5])
    mu = np.array([1, 1])
    covar = np.array([[1, -0.5],[-0.75, 1]])

    num_samp = 100
    mh_samp = np.zeros((num_samp,2))
    #M-H loop
    for i in range(2,num_samp):
        mh_samp[i,:] = mh_samp[i-1,:]
        x_star = np.random.multivariate_normal(mh_samp[i,:],np.eye(2))
        print(x_star)
        alpha = gaussianMD(np.array([x_star[0], mh_samp[i,1]]), mu, covar)/gaussianMD(mh_samp[i,:], mu, covar)
        if(random.random() < min(1, alpha)):
            mh_samp[i,0] = x_star[0]

        alpha = gaussianMD(np.array([mh_samp[i,0], x_star[1]]), mu, covar)/gaussianMD(mh_samp[i,:], mu, covar)
        if(random.random() < min(1, alpha)):
            mh_samp[i,1] = x_star[1]

    #Plot guassian
    xlim = [-2, 4]
    ylim = [-2, 4]  
    x = np.linspace(xlim[0],xlim[1],150)
    y = np.linspace(ylim[0],ylim[1],150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for (i,j), val in np.ndenumerate(X):
        x = np.array([X[i,j], Y[i,j]])
        Z[i,j] = gaussianMD(x, mu, covar)

    cmap = plt.cm.brg
    levels = 15
    ax.contour(X, Y, Z, levels, cmap=plt.cm.get_cmap(cmap, levels), linewidth=0.5, zorder=1)
    #Plot M-H
    ax.plot(mh_samp[:,0], mh_samp[:,1], 'x-k', alpha=0.75, linewidth='0.5', markersize='5.0', label='M-H Samples')

    ax.set_xlabel(r'$x_{1}$')
    ax.set_ylabel(r'$x_{2}$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()

    #====== Hamiltonian Monte-Carlo =====
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 4 Problem 3(c)', fontsize=14)

    #Guassian parameters
    x = np.array([0.5, 0.5])
    mu = np.array([1, 1])
    covar = np.array([[1, -0.5],[-0.75, 1]])

    num_samp = 100
    x_samp = np.zeros((num_samp,2))
    #HMC loop
    for i in range(2,num_samp):
        # Sample momentum
        p = np.random.normal(0,1,2)
        x_samp[i,:],p = leapFrog(x_samp[i-1,:], p, 50, 0.05, mu, covar)
        # M-H acceptance probability
        alpha = gaussianMD(np.array([x_samp[i,1]]), mu, covar)/gaussianMD(x_samp[i-1,:], mu, covar)
        if(random.random() > min(1, alpha)): #Greater than, so Reject 
            x_samp[i,:] = x_samp[i-1,:]

    #Plot guassian
    xlim = [-2, 4]
    ylim = [-2, 4]  
    x = np.linspace(xlim[0],xlim[1],150)
    y = np.linspace(ylim[0],ylim[1],150)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for (i,j), val in np.ndenumerate(X):
        x = np.array([X[i,j], Y[i,j]])
        Z[i,j] = gaussianMD(x, mu, covar)

    cmap = plt.cm.brg
    levels = 15
    ax.contour(X, Y, Z, levels, cmap=plt.cm.get_cmap(cmap, levels), linewidth=0.5, zorder=1)
    #Plot M-H
    ax.plot(x_samp[:,0], x_samp[:,1], 'x-k', alpha=0.75, linewidth='0.5', markersize='5.0', label='HMC Samples')

    ax.set_xlabel(r'$x_{1}$')
    ax.set_ylabel(r'$x_{2}$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()

    plt.show()