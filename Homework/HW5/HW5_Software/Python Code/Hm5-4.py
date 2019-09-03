'''
Nicholas Geneva
ngeneva@nd.edu
November 4, 2017
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

def particleFilter(nsteps, a, b, c, d):
    '''
    Completes a simple 1D linear Guassian particle filter
    xt = ax + N(0,b); yt = cxt + N(0,d);
    Args:
        nsteps (int) = number of steps to sample
        a, b, c, d (float) = linear guassian parameters
    Returns:
        s (np.array) = 2xD array of particle samples for x_t and y_t
    '''
    s = np.zeros((2,nsteps))
    s[0,0] = np.random.normal(0,1.0)
    for  i in range(1,nsteps):
        #P(x_t|x_t-1)
        s[0,i] = np.random.normal(a*s[0,i-1], np.sqrt(b))
        #P(y_t|x_t)
        s[1,i] = np.random.normal(c*s[0,i],  np.sqrt(d))

    return s

def kalmanFilter(nstep, Y, a, b, c, d):
    '''
    Completes a simple Kalman filter for 1D linear state-space
    xt = ax + N(0,b); yt = cxt + N(0,d);
    Args:
        nstep (int) = number of steps to sample
        Y (np.array) = 1xN set of observation data 
        a, b, c, d (float) = linear guassian parameters
    Returns:
       
    '''
    mu = np.zeros(nstep)
    var = np.zeros(nstep)
    mu[0] = 0
    var[0] = 1.0

    for  i in range(1,nstep):
        #Covar_{t|t-1}
        phat = a*var[i-1]*a + b
        k = phat*c/(c*phat*c + d)
        #Covar_{t|t}
        var[i] = phat - k*c*phat
        #mu_{t|t}
        mu[i] = a*mu[i-1] + k*(Y[i] - c*a*mu[i-1])

    return mu, var

def gaussian1D(x, mu, var):
    """
    Calculates 1D gaussian density
    Args:
        x (flost) = point of interest
        mu (float) = mean 
        var (float) = Variance squared 
    """
    small = 1e-8

    e = (x-mu)*(1/(var+small))
    e = e*(x-mu)
    e = np.exp(-0.5*e)

    return 1.0/(np.power(2*np.pi*(var+small),0.5))*e

def bootStrapFilter(y, nsteps, N, a, b, c, d, ess, resamp = 'standard'):
    '''
    Executes bootstrap filter
    Args:
        y (nd.array) = [D] array of observation data
        nsteps (int) = number of timesteps
        N (int) = number of particles
        a, b, c, d (float) = linear guassian parameters
        ess (float) = ESS trigger (set to number of particles if you want to resample every timestep)
        resamp (string) = resampling method (standard, systematic)
    Returns:
        x (nd.array) = [nsteps, D] array of states
        w_hist (nd.array) = [nsteps, D] array of filtering distributions g(y|x)
    '''
    small = 1e-8
    x = np.zeros((nsteps, N)) + small
    w_log = np.zeros((nsteps, N))
    w = np.zeros((nsteps, N))
    w_hist = np.zeros((nsteps, N))

    #Initialize x, weights, log-weights
    x[0,:] = np.random.normal(0, 1, N)
    w[0,:] = 1.0/N + np.zeros((N))
    w_log[0,:] = np.log(w[0,:])
    w_hist[0,:] = w[0,:]

    #Iterate over timesteps
    for i in range(1,nsteps):
        #First, sample particles for states
        x[i,:] = np.random.normal(a*x[i-1,:], np.sqrt(b), N)

        #Second update the importance weights
        w_hist[i,:] = gaussian1D(y[i], c*x[i,:], d)
        w_log[i,:] = w_log[i-1,:] + np.log(w_hist[i,:] + small)
        w_log[i,:] = w_log[i,:] - np.max(w_log[i,:])
        w[i,:] = np.exp(w_log[i,:])/np.sum(np.exp(w_log[i,:]))

        #Calculate Kish's effective sample size
        neff = 1.0/np.sum(np.power(w[i,:],2))
        #ESS trigger
        if(neff < ess):
            #Third resample the points
            if(resamp == 'systematic'):
                ind = resampleSystematic(w[i,:],N)
            else: #Standard resampling
                ind = resample(w[i,:],N)
            x = np.take(x, ind, 1)
            w[i,:] = 1.0/N + np.zeros((N))
            w_log[i,:] = np.log(w[i,:])

    return x, w_hist

def auxFilter(y, nsteps, N, a, b, c, d, ess = float("inf"), resamp = 'standard'):
    '''
    Executes fully adapted auxilary particle filter
    Args:
        y (nd.array) = [D] array of observation data
        nsteps (int) = number of timesteps
        N (int) = number of particles
        a, b, c, d (float) = linear guassian parameters
    Returns:
        x (nd.array) = [nsteps, D] array of states
        w_hist (nd.array) = [nsteps, D] array of filtering distributions g(y|x)
    '''
    small = 1e-5
    x = np.zeros((nsteps, N)) + small
    w_log = np.zeros((nsteps, N))
    w = np.zeros((nsteps, N))
    w_hist = np.zeros((nsteps, N))

    #Initialize x, weights, log-weights
    x[0,:] = np.random.normal(0, 1, N)
    w[0,:] = 1.0/N + np.zeros((N))
    w_log[0,:] = np.log(w[0,:])
    w_hist[0,:] = w[0,:]
    sig_n = 1.0/(np.sqrt(b)**-2+(np.sqrt(d)/c)**-2)

    #Iterate over timesteps
    for i in range(1,nsteps):
        #Now, sample particles for states
        mu_n = sig_n*(a*x[i-1,:]/b+c*y[i-1]/d)
        x[i,:] = np.random.normal(mu_n, np.sqrt(sig_n), N)

        #Update the importance weights
        w_hist[i,:] = gaussian1D(y[i], c*x[i,:], d)
        #w = P(y_{t+1}|x_{t+1})/P(y_{t+1}|mu_{t+1})
        w_log[i,:] = w_log[i-1,:] + np.log(w_hist[i,:] + small) + small
        w_log[i,:] = w_log[i,:] - np.max(w_log[i,:])
        w[i,:] = np.exp(w_log[i,:])/np.sum(np.exp(w_log[i,:]))

        #Calculate Kish's effective sample size
        neff = 1.0/np.sum(np.power(w[i,:],2))
        #ESS trigger
        if(neff < ess):
            #Third resample the points
            if(resamp == 'systematic'):
                ind = resampleSystematic(w[i,:],N)
            else: #Standard resampling
                ind = resample(w[i,:],N)
            x = np.take(x, ind, 1)
            w[i,:] = 1.0/N + np.zeros((N))
            w_log[i,:] = np.log(w[i,:])

    return x, w_hist

def resample(weight, N):
    '''
    Execuates the standard resampling method
    Params:
        weight (nd.array) = 1xN array of weights for each particle
        N (int) = number of particles
    Returns:
        y (nd.array) = 1xN array of indexes of particles to keep
    '''
    w_sort = np.sort(weight)
    w_ind = np.argsort(weight)
    y = np.argsort(weight)

    for i in range(0, N):
        u = np.random.uniform()
        t = 0
        for k in range(0, N):
            t = t + w_sort[k]
            if (u<t):
                y[i] = w_ind[k]
                break
    
    return y

def resampleSystematic(weight, N):
    '''
    Execuates the systematic resampling method
    Params:
        weight (nd.array) = 1xN array of weights (Particle values here)
        N (int) = number of wieghts or particles
    Returns:
        y (nd.array) = 1xN array of indexes of particles to keep
    '''
    w_sort = np.sort(weight)
    w_ind = np.argsort(weight)
    y = np.argsort(weight)

    #Set up the partitions
    w_part = np.array([float(i)/N for i in range(0,N)])
    #Now sample the little pertabation
    u = np.random.uniform(0,1.0/N)
    #Add pertabation to partitions
    w_part = w_part + u
    #Now resample
    for i in range(0, N):
        t = 0
        for k in range(0, N):
            t = t + w_sort[k]
            if (w_part[i]<t):
                y[i] = w_ind[k]
                break
    
    return y

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)
    #========== Problem 4 (a) ==========
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 5 Problem 4(a)', fontsize=14)

    N = 2000 #Number of particles
    Yt = particleFilter(N, a = 0.7, b = 0.1, c = 0.5, d = 0.1)
    emp_mu = np.sum(Yt[1,:])/N
    emp_var = np.sum(np.power(Yt[1,:]-emp_mu, 2))/N
    x = range(N)
    ax.plot(x,Yt[0,:],linestyle='-', color = 'red', linewidth=1.0, label='States')
    ax.plot(x,Yt[1,:], 'o', markersize=3.5, label='Observations')
    ax.set_xlabel('t')
    ax.legend()
    
    #========== Problem 4 (b) ==========
    mu, var = kalmanFilter(N, Yt[1,:], a = 0.7, b = 0.1, c = 0.5, d = 0.1)
    x = np.linspace(-2.0,2.0,150)
    y = np.zeros(x.shape[0])
    for i, x0 in enumerate(x):
        y[i] = sc.stats.norm.pdf(x0, mu[-1], np.sqrt(var[-1]))
    
    #Tracking
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    f.suptitle('Homework 5 Problem 4 (b)', fontsize=14)
    x = range(N)
    ax.plot(x, Yt[0,:], linestyle='-', color = 'red', linewidth=1.0, label='True States')
    ax.plot(x, mu, linestyle='-', color = 'green', linewidth=1.0, label='Kalman Filter Mean')
    ax.set_xlim([1900, 2000])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()

    #========== Problem 4 (c) ==========
    Np = 100 #Number of particles
    bs_x, bs_whist = bootStrapFilter(Yt[1,:], N, Np, a = 0.7, b = 0.1, c = 0.5, d = 0.1, ess = Np)
    bsMu = [sum(bs_x[i,:]) / float(Np) for i in range(N)]
    bsVar = [np.sum(np.power(bs_x[i,:] - bsMu[i],2))/float(Np) for i in range(N)]

    x = np.linspace(-2.0,2.0,150)
    y = np.zeros(x.shape[0])
    for i, x0 in enumerate(x):
        y[i] = sc.stats.norm.pdf(x0, mu[-1], np.sqrt(var[-1]))

    #Tracking
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    f.suptitle('Homework 5 Problem 4(c)', fontsize=14)
    x = range(N)
    ax.plot(x, Yt[0,:], linestyle='-', color = 'red', linewidth=1.0, label='True States')
    ax.plot(x, mu, linestyle='-', color = 'green', linewidth=1.0, label='Kalman Filter Mean')
    ax.plot(x, bsMu, linestyle='-', color = 'blue', linewidth=1.0, label='Bootstrap Mean')
    ax.set_xlim([1900, 2000])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()

    #Error plots
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    f.suptitle('Homework 5 Problem 4(c)', fontsize=14)
    x = range(N)
    ax.plot(x, np.abs(bsMu-mu), linestyle='-', color = 'red', linewidth=1.0, label='Mean Error')
    ax.plot(x, np.abs(bsVar-var), linestyle='-', color = 'blue', linewidth=1.0, label='Variance Error')
    ax.set_xlim([1900, 2000])
    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.legend()

    #========== Problem 4 (d) ==========
    Np = 100 #Number of particles
    aux_x, aux_whist = auxFilter(Yt[1,:], N, Np, a = 0.7, b = 0.1, c = 0.5, d = 0.1)
    auxMu = [sum(aux_x[i,:]) / float(Np) for i in range(N)]
    auxVar = [np.sum(np.power(aux_x[i,:] - auxMu[i],2))/float(Np) for i in range(N)]

    x = np.linspace(-2.0,2.0,150)
    y = np.zeros(x.shape[0])
    for i, x0 in enumerate(x):
        y[i] = sc.stats.norm.pdf(x0, mu[-1], np.sqrt(var[-1]))

    #Tracking
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    f.suptitle('Homework 5 Problem 4(d)', fontsize=14)
    x = range(N)
    ax.plot(x, Yt[0,:], linestyle='-', color = 'red', linewidth=1.0, label='True States')
    ax.plot(x, mu, linestyle='-', color = 'green', linewidth=1.0, label='Kalman Filter Mean')
    ax.plot(x, bsMu, linestyle='-', color = 'blue', linewidth=1.0, label='Bootstrap Mean')
    ax.plot(x, auxMu, linestyle='-', color = 'orange', linewidth=1.0, label='Auxilary Mean')
    ax.set_xlim([1900, 2000])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()

    #Error plots
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    f.suptitle('Homework 5 Problem 4(d)', fontsize=14)
    x = range(N)
    ax.plot(x, np.abs(auxMu-mu), linestyle='--', color = 'red', linewidth=1.0, label='Aux Mean Error')
    ax.plot(x, np.abs(auxVar-var), linestyle='--', color = 'blue', linewidth=1.0, label='Aux Variance Error')
    ax.plot(x, np.abs(bsMu-mu), linestyle='-', color = 'red', linewidth=1.0, label='Boot Mean Error')
    ax.plot(x, np.abs(bsVar-var), linestyle='-', color = 'blue', linewidth=1.0, label='Boot Variance Error')
    ax.set_xlim([1900, 2000])
    ax.set_xlabel('t')
    ax.set_ylabel('Error')
    ax.legend()
    
    #========== Problem 4 (e) ==========
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    f.suptitle('Homework 5 Problem 4(e)', fontsize=14)

    t = range(1950,N)
    for i in range(1,Np):
        ax.plot(t, aux_x[1950:,i], linewidth=1.0)

    #========== Problem 4 (f) ==========
    Np = 100 #Number of particles
    sys_x, sys_whist = auxFilter(Yt[1,:], N, Np, a = 0.7, b = 0.1, c = 0.5, d = 0.1, ess = Np, resamp = 'systematic')
    sysMu = [sum(sys_x[i,:]) / float(Np) for i in range(N)]
    sysVar = [np.sum(np.power(sys_x[i,:] - sysMu[i],2))/float(Np) for i in range(N)]

    x = np.linspace(-2.0,2.0,150)
    y = np.zeros(x.shape[0])
    for i, x0 in enumerate(x):
        y[i] = sc.stats.norm.pdf(x0, mu[-1], np.sqrt(var[-1]))

    #Tracking
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    f.suptitle('Homework 5 Problem 4(f)', fontsize=14)
    x = range(N)
    ax.plot(x, Yt[0,:], linestyle='-', color = 'red', linewidth=1.0, label='True States')
    ax.plot(x, mu, linestyle='-', color = 'green', linewidth=1.0, label='Kalman Filter Mean')
    ax.plot(x, bsMu, linestyle='-', color = 'blue', linewidth=1.0, label='Bootstrap Mean')
    ax.plot(x, sysMu, linestyle='-', color = 'orange', linewidth=1.0, label='Bootstrap Mean (Systematic)')
    ax.set_xlim([1900, 2000])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()

    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(10, 3))
    f.suptitle('Homework 5 Problem 4(f)', fontsize=14)

    t = range(1950,N)
    for i in range(1,Np):
        ax.plot(t, sys_x[1950:,i], linewidth=1.0)

    #========== Problem 4 (g) ==========
    Np = 100 #Number of particles
    ess_x, ess_whist = auxFilter(Yt[1,:], N, Np, a = 0.7, b = 0.1, c = 0.5, d = 0.1, ess = 0.5*Np, resamp = 'systematic')
    essMu = [sum(sys_x[i,:]) / float(Np) for i in range(N)]
    essVar = [np.sum(np.power(ess_x[i,:] - essMu[i],2))/float(Np) for i in range(N)]

    x = np.linspace(-2.0,2.0,150)
    y = np.zeros(x.shape[0])
    for i, x0 in enumerate(x):
        y[i] = sc.stats.norm.pdf(x0, mu[-1], np.sqrt(var[-1]))

    #Tracking
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    f.suptitle('Homework 5 Problem 4(g)', fontsize=14)
    x = range(N)
    ax.plot(x, Yt[0,:], linestyle='-', color = 'red', linewidth=1.0, label='True States')
    ax.plot(x, mu, linestyle='-', color = 'green', linewidth=1.0, label='Kalman Filter Mean')
    ax.plot(x, bsMu, linestyle='-', color = 'blue', linewidth=1.0, label='Bootstrap Mean (ESS = Np)')
    ax.plot(x, essMu, linestyle='-', color = 'orange', linewidth=1.0, label='Bootstrap Mean (ESS = 0.5*Np)')
    ax.set_xlim([1900, 2000])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()

    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(10, 3))
    f.suptitle('Homework 5 Problem 4(g)', fontsize=14)

    t = range(1950,N)
    for i in range(1,Np):
        ax.plot(t, sys_x[1950:,i], linewidth=1.0)

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()