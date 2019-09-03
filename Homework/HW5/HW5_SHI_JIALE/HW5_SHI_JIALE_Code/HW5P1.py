
# coding: utf-8

# In[84]:


import sys
import random
import numpy as np
import scipy.stats
import matplotlib as mlp
import matplotlib.pyplot as plt

#from matplotlib import rc
#from scipy import special


x = np.linspace(0,8,10000)

def f(v):
    return 2*np.sin(np.pi/1.5*v)

def tarfun(v):
    return pow(v,0.65)*np.exp(-v*v/2)

def profun(v):
    return scipy.stats.gamma.pdf(v,a=3,scale=1/2.5)


numSamples = 1000000
#Accept - Reject algorithm, Sample from laplacian
def Importance_sampling():
    samples = np.random.gamma(3,1/2.5,numSamples)
    weight = tarfun(samples)/profun(samples)
    return samples, weight

samps,wt = Importance_sampling()

plt.figure(figsize=(8,6))
plt.plot(x,tarfun(x),label ="target")
plt.plot(x,profun(x),label ="proposal")
plt.hist(samps,bins=100, alpha=0.4, label=u'sampled histogram', normed=True) 
#plt.plot(x,profun(x),label ="proposal Gamma(3,2.5)")
plt.legend(fontsize ='x-large')
plt.xlabel("x",size=20)
plt.ylabel("p(x)",size = 20)
#plt.xlim(0,8)
#plt.ylim(0,1)
plt.show()


# In[85]:


plt.figure(figsize=(8,6))
plt.scatter(samps,wt,label ="weight")
#plt.plot(x,profun(x),label ="proposal Gamma(3,2.5)")
plt.legend(fontsize ='x-large')
plt.xlabel("samps",size=20)
plt.ylabel("weight",size = 20)
#plt.xlim(0,8)
#plt.ylim(0,1)
plt.show()


# In[86]:


E = 0.0
for i in range(0,numSamples):
    E = E+wt[i]*f(samps[i])

E = E/numSamples
print (E)


# In[88]:


plt.figure(figsize=(8,6))
#plt.plot(x,tarfun(x),label ="target")
plt.plot(x,tarfun(x)/profun(x),label ="proposal")

plt.legend(fontsize ='x-large')
plt.xlabel("x",size=20)
plt.ylabel("p(x)",size = 20)
#plt.xlim(0,8)
plt.ylim(0,1)
plt.show()

