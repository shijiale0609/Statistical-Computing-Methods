'''
Jiale Shi
Statistical Computing for Scientists and Engineers
Homework 1 5c
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
    lmu = 0
    for i in range(0,N):
        lmu = lmu+np.log(x[i])
    lmu = lmu/N
    for i in range(0,N):
        sig2 = sig2+pow(x[i]-mu,2)
    sig2 = sig2/N
    print ("mu:", mu)
    print ("log(mu)", np.log(mu))
    print ("lmu",lmu)
    return mu, np.sqrt(sig2)


if __name__== "__main__":
    
    # Start by reading in the camera data
    data = readCSV()
    # Create histogram
    weight = np.histogram(data[:,3], bins=9)
    # Get x/y training points (for x we take the center of the bins)
    x0 = 0.5*(weight[1][:-1]+weight[1][1:])
    y0 = weight[0]
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
    bins = weight[1][:]
    y_norm = np.sum((bins[1:]-bins[:-1])*y0) # Finite Integral
    
    plt.bar(x0, y0/y_norm, align='center',width=200.0, edgecolor="k",label="Histogram")
    # Plot MLE Guassian
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma),'r',label="MLE Guassian")
    plt.plot(x, scipy.stats.gamma.pdf(x, a=1.84, scale=166.6387422858897 ),'green',label="MLE Gamma")  
    plt.title('Camera Weight')
    plt.xlabel('Camera Weight')
    plt.legend()
    
    # f1.savefig('Hm1-P5a-Res.png', bbox_inches='tight')

    plt.show()

