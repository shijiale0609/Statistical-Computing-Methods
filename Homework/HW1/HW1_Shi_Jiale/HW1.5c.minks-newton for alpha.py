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
from scipy.special import polygamma,digamma

a = np.array([1.0]*50)

for i in range(0,49):
    a[i+1]= 1/(1/a[i]+(5.429821356855465-digamma(a[i])-5.725593821052867+np.log(a[i]))/(-pow(a[i],2)*polygamma(1,a[i])))


# In[7]:


b = np.array([1.0]*50)
for i in range(0,50):
    b[i]= i


# In[21]:


plt.plot(b,a,label = "minka-newton method")
plt.plot(b,a-a+1.84,label="alpha = 1.84")
plt.xlabel("iteration")
plt.ylabel("alpha")
plt.legend()
#plt.ylim(-0.1,0.1)
plt.show()


# In[16]:


alpha = 1.84
beta = 306.61528580603704/alpha
print (alpha, beta)

