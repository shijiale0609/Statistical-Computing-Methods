'''
Nicholas Geneva
ngeneva@nd.edu
October 28, 2017
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

def readFileData(fileName):
    '''
    Reads in break data from files
    @params:
        fileName(string): File name
    Returns:
        data (np.array): array of data read from file
    '''
    #Attempt to read text file and extact data into a list
    try:
        file_object  = open(str(fileName), "r").read().splitlines()[:]
        #seperate columns and strip white space
        data_list = [float(my_string.strip()) for my_string in file_object] 
    except OSError as err:
        print("OS error: {0}".format(err))
        return
    except IOError as err:
        print("File read error: {0}".format(err))
        return
    except:
        print("Unexpected error:{0}".format(sys.exc_info()[0]))
        return

    data = np.asarray(data_list)
    return data

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

    return 1.0/(np.power(2*np.pi*var,0.5))*e

def resample(weight, N):
    '''
    Execuates the standard multinomal resampling method
    Useful Reference: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1521264
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

def bootStrapFilter(y, nsteps, N, phi = 0.98, sigma = 0.16, beta = 0.70, ess = float("inf"), resamp = 'standard'):
    '''
    Executes bootstrap filter
    Args:
        y (nd.array) = [D] array of observation data
        nsteps (int) = number of timesteps
        N (int) = number of particles
        phi, sigma, beta (float) = hyperparameters
        eff (float) = ESS trigger (default is inf, so resample every timestep)
        resamp (string) = resampling method (standard, systematic)
    Returns:
        x (nd.array) = [nsteps, D] array of states
        w_hist (nd.array) = [nsteps, D] array of filtering distributions g(y|x)
    '''
    small = 1e-5
    x = np.zeros((nsteps, N)) + small
    w_log = np.zeros((nsteps, N)) + np.log(1.0/N)
    w = np.zeros((nsteps, N))
    w_hist = np.zeros((nsteps, N))

    #Initialize x, weights, log-weights
    x[0,:] = np.random.normal(phi*x[0,:], 2*sigma, N) #Initialize on a normal with 0 mean
    w_log[0,:] = np.log(gaussian1D(y[0], 0, beta*beta*np.exp(x[0,:])) + small)
    w_log[0,:] = w_log[0,:] - np.max(w_log[0,:])
    w[0,:] = np.exp(w_log[0,:])/np.sum(np.exp(w_log[0,:]))
    w_hist[0,:] = gaussian1D(y[0], 0, beta*beta*np.exp(x[0,:]))

    #Iterate over timesteps
    for i in range(1,nsteps):
        #First, sample particles for states
        x[i,:] = np.random.normal(phi*x[i-1,:], sigma, N)

        #Second update the importance weights
        w_log[i,:] = w_log[i-1,:] + np.log(gaussian1D(y[i], 0, beta*beta*np.exp(x[i,:])) + small)
        w_hist[i,:] = np.exp(w_log[i,:] - w_log[i-1,:])
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

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #========== Read in data from file ==========
    y = readFileData("logreturns2012to2014.csv")

    #========== Problem 1 (a) ==========
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 5 Problem 1(a)', fontsize=14)

    #Params
    nsteps = 500 #Timesteps
    N = 40 #Particles
    phi = 0.98; sigma = 0.16; beta = 0.70

    x, w_hist = bootStrapFilter(y, nsteps, N, 0.98, 0.16, 0.70)

    t = range(0,nsteps)
    ax.plot(t, np.sum(x[:,:],1)/N, 'k', label='x mean')
    ax.plot(t, y[:nsteps], 'o', markersize=3.5, label='observation')
    ax.set_xlabel('t')
    ax.legend()
    
    #========== Problem 1 (b) ==========
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 5 Problem 1(b)', fontsize=14)

    #Params
    N = 20
    nsteps = 500 #Timesteps
    B = 10 #Number of betas
    beta_like = np.zeros((10,B))

    for j, beta in enumerate(np.linspace(0.25,2,B)):
        for i in range(10):
            #Bootstrap filter
            x, w_hist = bootStrapFilter(y, nsteps, N, 0.98, 0.16, beta)
            
            #Commute log marginal likelyhood
            w_sum = np.sum(w_hist, 1) + 1e-5
            beta_like[i,j] = np.sum(np.log(w_sum) - np.log(N))

    ax.boxplot(beta_like)
    ax.set_xticklabels(["%.2f" % num for num in np.linspace(0.25,2,B)])
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$log(p)$')

    #========== Problem 1 (c) ==========
    #Set up subplots
    f, ax = plt.subplots(2, 2, figsize=(9, 9))
    f.suptitle('Homework 5 Problem 1(c)', fontsize=14)

    #Params
    nsteps = 500 #Timesteps
    N = 10 #Particles
    phi = 0.98; sigma = 0.16; beta = 0.70

    x, w_hist = bootStrapFilter(y, nsteps, N, 0.98, 0.16, 0.70)
    ess_x, ess_w_hist = bootStrapFilter(y, nsteps, N, 0.98, 0.16, 0.70, 0.5*N)
    syst_x, syst_w_hist = bootStrapFilter(y, nsteps, N, 0.98, 0.16, 0.70, N, resamp = 'systematic')
    syst_x2, syst_w_hist2 = bootStrapFilter(y, nsteps, N, 0.98, 0.16, 0.70, 0.5*N, resamp = 'systematic')

    t = range(0,nsteps)
    ax[0,0].set_title("Multinomial Resampling [ESS = Np]")
    for i in range(1,N):
        ax[0,0].plot(t, x[:,i], linewidth=1.0)

    ax[0,1].set_title("Multinomial Resampling [ESS = 0.5*Np]")
    for i in range(1,N):
        ax[0,1].plot(t, ess_x[:,i], linewidth=1.0)

    ax[1,0].set_title("Systematic Resampling [ESS = Np]")
    for i in range(1,N):
        ax[1,0].plot(t, syst_x[:,i], linewidth=1.0)

    ax[1,1].set_title("Systematic Resampling [ESS = 0.5*Np]")
    for i in range(1,N):
        ax[1,1].plot(t, syst_x2[:,i], linewidth=1.0)

    for ax0 in ax.reshape(-1):
        ax0.set_xlim([400, 500])
        ax0.set_xlabel('t')
        ax0.set_xlabel('x')

    plt.tight_layout(rect=[0,0, 1.0, 0.93])
    plt.show()