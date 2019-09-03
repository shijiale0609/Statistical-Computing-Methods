
# coding: utf-8

# In[2]:


'''
Statistical Computing for Scientists and Engineers
Homework 4 Problem 3 b1
Fall 2018
University of Notre Dame
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats

# the true distribution
def f(v):
    return scipy.stats.gamma.pdf(v,a=4.3,scale=1/6.2)
# the proposal distribution   
def q(v):
    return scipy.stats.gamma.pdf(v,a=5,scale=1/6)

#Initialization
numSamples = 50000

samp= np.zeros(numSamples)
samp[0]=5
#Accept - Reject algorithm, Sample from laplacian
#def accept_reject():
#    samples = np.random.gamma(4,1/7,numSamples)
#    acceptanceProb = f(samples)/(M*q(samples))
#    unif_samp = np.random.rand(1,numSamples)
#    accepted = unif_samp < acceptanceProb
#    return samples, accepted, unif_samp

# all the samps, accepted, unif_samps
#samps,accepteds,unif_samps = accept_reject()

for i in range(1, numSamples):
    y = scipy.stats.gamma.rvs(5,0,scale=1/6);        
    prob = min(1, q(samp[i-1])/q(y)*(f(y)/f(samp[i-1])));
    u = np.random.uniform()
    if ( u <= prob): 
        samp[i] = y; 
    else:
        samp[i] = samp[i-1];


#calculate the expectation
E = np.array([0.0]*numSamples)
Sum = 0
List = np.array([0]*numSamples)
for i in range(0,numSamples):
        Sum= Sum+samp[i];
        E[i]=Sum/(i+1);
        List[i]=i+1

#plot the expectation
plt.figure(figsize=(8,8))
plt.plot(List,E)
plt.ylabel("<E>")
plt.xlabel("Iteration")
plt.savefig('h4p3b21.png')
plt.show()

#calculate the convergence
cov = np.array([0.0]*numSamples)
for i in range(0,numSamples):
    cov[i]=np.mean(pow(E[0:i]-samp[0:i],2))
    
#plot the covergence
plt.figure(figsize=(8,8))
plt.plot(List,cov)
plt.ylabel("COV")
plt.xlabel("Iteration")
plt.savefig('h4p3b22.png')
plt.show()

x = np.linspace(0,10,100000)
# plot histogram & true distribution
plt.figure(figsize=(8,8))
plt.hist(samp,bins=100, alpha=0.4, label=u'sampled histogram', normed=True) 
plt.plot(x, f(x), 'r', label=u'True distribution') # f(x) is the True distribution
plt.legend()
plt.xlim([0,8])
plt.savefig('h4p3b23.png')
plt.show()


# In[4]:


f=open("/Users/shijiale1995/ecovb2.txt","a+")
for i in range(0,numSamples):
    f.write(str(E[i]))
    f.write(" ")
    f.write(str(cov[i]))
    f.write("\n")
f.close()

