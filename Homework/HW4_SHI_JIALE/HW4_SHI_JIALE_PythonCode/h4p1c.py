
# coding: utf-8

# In[3]:


'''
Statistical Computing for Scientists and Engineers
Homework 4 Problem 1 c
Fall 2018
University of Notre Dame
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats

# the true distribution
def f(v):
    return 1/np.sqrt(2*np.pi)*np.exp(-v*v/2)
# the proposal distribution   
def q(v):
    return np.exp(-2*abs(v))

# supremum of f/q
M = 0.5*np.sqrt(2/np.pi)*np.exp(2)

#Initialization
numSamples = 10000

x= np.linspace(-4,4,1001)

#Accept - Reject algorithm, Sample from laplacian
def accept_reject():
    samples = np.random.laplace(0,0.5,numSamples)
    acceptanceProb = f(samples)/(M*q(samples))
    unif_samp = np.random.rand(1,numSamples)
    accepted = unif_samp < acceptanceProb
    return samples, accepted, unif_samp

# all the samps, accepted, unif_samps
samps,accepteds,unif_samps = accept_reject()

#filter the accepted samps
Samp= np.array([])
Unif_samp = np.array([])
SumN = 0
for i in range(0,numSamples):
    if (accepteds[0][i]==1):
        Samp=np.append(Samp,[samps[i]])
        Unif_samp = np.append(Unif_samp,[unif_samps[0][i]])
        SumN=SumN+1 # SumN counts the accepted samps 

#calculate the convergence
E = np.array([0.0]*SumN)
Sum = 0
List = np.array([0]*SumN)
for i in range(0,SumN):
        Sum= Sum+Samp[i];
        E[i]=Sum/(i+1);
        List[i]=i+1

#plot the convergence
plt.figure(figsize=(8,8))
plt.plot(List,E)
plt.ylabel("<E>")
plt.xlabel("Iteration")
plt.savefig('h4p1c1.png')
plt.show()

cov = np.array([0.0]*SumN)
for i in range(0,SumN):
    cov[i]=np.mean(pow(E[0:i]-Samp[0:i],2))
    
#plot the covergence
plt.figure(figsize=(8,8))
plt.plot(List,cov)
plt.ylabel("COV")
plt.xlabel("Iteration")
plt.savefig('h4p1c2.png')
plt.show()



#plot the True distribution & proposal distribution & accepted samps
plt.figure(figsize=(8,8))
plt.plot(x,M*q(x),c='orange',label=u'Proposal distribution')
plt.plot(x, f(x), 'r', label=u'True distribution') # f(x) is the True distribution
plt.scatter(Samp,M*Unif_samp*q(Samp),s=2,c='blue')
plt.legend()
plt.xlim([-4,4])
plt.savefig('h4p1c3.png')
plt.show()


# plot histogram & true distribution
plt.figure(figsize=(8,8))
plt.hist(Samp,bins=50, alpha=0.4, label=u'sampled histogram', normed=True) 
plt.plot(x, f(x), 'r', label=u'True distribution') # f(x) is the True distribution
plt.legend()
plt.xlim([-4,4])
plt.savefig('h4p1c4.png')
plt.show()

