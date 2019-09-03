'''
Jiale Shi 
Statistical Computing for Scientists and Engineers
Homework 1 Problem 5d
Fall 2018
University of Notre Dame
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import errno
import os.path

def readCSV(fileDir='.'):
    '''
    Reads in .csv data file
    @args: fileDir = path to file
    '''
    file_path = os.path.join("/Users/shijiale1995/Downloads", 'camera.csv')
    
    if(os.path.exists(file_path)):
        return np.genfromtxt(file_path, delimiter=',',skip_header=2)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

def mleGuassian2D(x):
    '''
    Does MLE for a two deminsional guassian
    @args: x = np.array([2xN]) training data
    @returns: mu, sigma = mean [2x1] and std [2x2] of trained guassian
    '''
    N = x.shape[1]
    mu = np.zeros((2,1))
    sig2 = np.zeros((2,2))
    print (mu)
    print(sig2)
    t = np.zeros((2,1))
    for i in range(0,N):
        t[0][0] = x[0][i]
        t[1][0] = x[1][i]
        mu = mu+t 
    mu = mu/N
    print(mu)
    for i in range(0,N):
        t[0][0] = x[0][i]-mu[0][0]
        t[1][0] = x[1][i]-mu[1][0]
        sig2 = sig2+t*t.T
      
    sig2 = sig2/N
   
    print (mu)
    print (sig2)
    return mu, np.sqrt(sig2)

def getGuassianDist(mu0, covar0, X, Y):
    '''
    Used to plot Gaussian contour
    @args: mu0 = np.array([Dx1]) vector of means, 
           covar0 = np.array([DxD]) convariance array,
            X, Y = [NxM] mesh array with points to eval Gaussian at
    @returns: Z = [NxM] array of the contour field
    '''
    Z = np.zeros(X.shape)
    #Get guassian distribution
    g_reg = 1.0/(2.0*np.pi) *1/(np.linalg.det(covar0)**(0.5))
    covar_i = np.linalg.inv(covar0)
    for (i,j), val in np.ndenumerate(X):
        x = np.expand_dims(np.array([X[i,j], Y[i,j]]), axis=1)
        Z[i,j] = g_reg*np.exp(-0.5*((x-mu0).T.dot(covar_i)).dot((x-mu0)))

    return Z


if __name__== "__main__":
    
    # Start by reading in the camera data
    data = readCSV()
    # Get data
    x0 = np.stack([data[:,2], data[:,1]], axis=0)
    # MLE
    mu, sigma = mleGuassian2D(x0)

    print('Plotting Figure')
    # Plot Normalized Histogram
    plt.scatter(x0[0], x0[1], color='k', marker='x',label="Training Data")
    # Plot MLE Guassian
    x = np.linspace(min(x0[0])-100,max(x0[0])+100,150)
    y = np.linspace(min(x0[1])-5,max(x0[1])+5,150)
    X, Y = np.meshgrid(x, y)
    Z = getGuassianDist(mu, np.power(sigma,2), X, Y)
    #Plot guassian
    cmap = plt.cm.brg
    levels = 15
    plt.contour(X, Y, Z, levels, cmap=plt.cm.get_cmap(cmap, levels), zorder=1)

    plt.title('Camera Resolution vs. Year')
    plt.xlabel('Camera Resolution Width (Pixels)')
    plt.ylabel('Camera Release Year')
    plt.legend()
    
    # plt.savefig('Hm1-P5d.png', bbox_inches='tight')
    plt.show()

