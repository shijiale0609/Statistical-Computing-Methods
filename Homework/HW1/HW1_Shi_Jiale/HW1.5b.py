'''
Jiale Shi
Statistical Computing for Scientists and Engineers
Homework 1 5b
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

def mleGuassian(x):
    '''
    Does MLE for a single deminsional guassian
    @args: x = training data
    @returns: mu, sigma = mean and std of trained guassian
    '''
    N = x.shape[0]
    mu = 0
    sig2 = 0
    print (N)
    for i in range(0,N):
        mu = mu+x[i]
    mu= mu/N
    for i in range(0,N):
        
        sig2 = sig2+pow(x[i]-mu,2)
    sig2 = sig2/N
    
    return mu, np.sqrt(sig2)


if __name__== "__main__":
    
    # Start by reading in the camera data
    data = readCSV()
    # Create histogram
    max_res = np.histogram((data[:,2]-2491.7618497109806)/752.5256308818603, bins=20)
    # Get x/y training points (for x we take the center of the bins)
    x0 = 0.5*(max_res[1][:-1]+max_res[1][1:])
    y0 = max_res[0]
    # Set up observations
    x_data = []
    for i, val in enumerate(x0):
        x_data.append(np.repeat(val, y0[i]))
    x_data = np.concatenate(x_data,axis=0)

    # MLE
    mu, sigma = mleGuassian(x_data)
    print (x_data)
    print (mu,sigma)
    print('Plotting Figure')
    f1 = plt.figure(1)
    # Plot Normalized Histogram
    bins = max_res[1][:]
    y_norm = np.sum((bins[1:]-bins[:-1])*y0) # Finite Integral
    
    plt.bar(x0, y0/y_norm, align='center',width=200/752.5256308818603, edgecolor="k",label="Histogram")
    # Plot MLE Guassian
    x = np.linspace(mu - 4*sigma, mu + 4*sigma,200)
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma),'r',label="MLE Guassian")  
    plt.title('Camera Max Resolution')
    plt.xlabel('Camera Max Resolution (Pixels)')
    plt.legend()
   
    # f1.savefig('Hm1-P5a-Res.png', bbox_inches='tight')

    plt.show()

