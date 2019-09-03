'''
Statistical Computing for Scientists and Engineers
Homework 4
Fall 2018
University of Notre Dame
'''
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gamma
import scipy as sc


def metropolis_hastings(#fill the parameters):
########## add code below ##################



######## add code above ################
    return samples, accepted


samps,accepted = accept_reject(#fill the parameters)
#Plot the convergence
########## add code below ##################

plt.savefig('Convergence_Problem-3b.png')
######## add code above ################
# plot   histogram
plt.hist(samps,bins=100, alpha=0.4, label=u'sampled histogram', normed=True) 

# plot the True distribution
x= np.linspace(0,2,100)
plt.plot(x, f(x), 'r', label=u'True distribution') # f(x) is the True distribution
plt.legend()
plt.xlim([0,2])
plt.savefig('Problem-3b.png')



        



