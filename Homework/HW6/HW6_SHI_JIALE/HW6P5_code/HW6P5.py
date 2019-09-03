#Jiale Shi
#Statistic Computing
#Final HW 
#Problem 5

import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt
from numpy.random import random

numSamples = 100
np.random.seed(100)
#randomly generate 100 particles xi from Gaussian(0,1) distribution
samples = scipy.stats.norm.rvs(loc=0,scale=1,size = numSamples)

# 100 weights wi
weight = scipy.stats.norm.pdf(samples,loc=0,scale=1)

# normalize the weights
weight = weight/np.sum(weight)

#use the estimate mean m of Gaussian(0,1)distribution
mean = np.sum(samples*weight)
#print ("m:", mean)
#print (weight)

# multinomial resampling
def multinomial_resampling(w,s):
    N = len(w)
    cumulative_sum = np.cumsum(weight);
    indexes = np.zeros(N,'i');
    i = 0;
    while i<N:
        sampl = random();
        j = 1;
        while cumulative_sum[j] < sampl:
            j = j+1;
        indexes[i] = j
        i = i+1;
    resamples = []   
    #print (indexes)
    for index,item in enumerate(indexes):
         resamples.extend([s[item]])
    mean= np.mean(resamples)
    return mean


# systematic resampling
def systematic_resampling(w,s):
    #np.random.seed(100)
    N = len(w)
    cumulative_sum = np.cumsum(w)
    positions = (random() + np.arange(N)) / N
    #print (positions)
    indexes = np.zeros(N,'i')
    
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    resamples = [] 
    #print (indexes)
    #print (indexes)
    for index,item in enumerate(indexes):
         resamples.extend([s[item]])
    mean= np.mean(resamples)
    return mean


#stratified resampling
def stratified_resampling(w,s):
    #np.random.seed(100)
    N = len(w)
    
    cumulative_sum = np.cumsum(w)
    positions = (random(N) + np.arange(N))/N
    indexes = np.zeros(N,'i')
    
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    resamples = []  
    #print (indexes)
    for index,item in enumerate(indexes):
         resamples.extend([s[item]])
    mean= np.mean(resamples)
    return  mean

m_m = []
m_s = []
m_t = []
for i in range(1,1000):
        #m_m.extend([multinomial_resampling(weight,samples)])
        m_s.extend([systematic_resampling(weight,samples)])
        m_m.extend([multinomial_resampling(weight,samples)])
        m_t.extend([stratified_resampling(weight,samples)])
print ("Bingo!")


fig,ax = plt.subplots(figsize=(8,8))
plt.plot(np.arange(1,1000),mean-m_m,label="m-mm")
plt.plot(np.arange(1,1000),mean-m_s,label="m-ms")
plt.plot(np.arange(1,1000),mean-m_t,label="m-mt")
plt.xlim(0,1000)
plt.legend(fontsize =20)
plt.ylabel("$m-m_{i}$",size =36)
plt.xlabel("Iteration",size =36)
for side in ax.spines.keys():
     ax.spines[side].set_linewidth(3)
plt.show()


dmm = mean - m_m
dms = mean - m_s
dmt = mean - m_t

V_m = np.var(dmm)
V_s = np.var(dms)
V_t = np.var(dmt)
print ("the variance of m-mm:", V_m)
print ("the variance of m-ms:", V_s)
print ("the variance of m-mt:", V_t)



