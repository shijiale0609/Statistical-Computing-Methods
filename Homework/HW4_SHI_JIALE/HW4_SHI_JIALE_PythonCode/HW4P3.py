
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "/Users/shijiale1995"
G1 = pd.read_csv("{}/ecovb1.txt".format(path),delimiter=" ")
G2 = pd.read_csv("{}/ecovb2.txt".format(path),delimiter=" ")


# In[2]:


numSamples = 50000
List = np.array([0]*numSamples)
for i in range(0,numSamples):
        List[i]=i+1


# In[11]:


plt.plot(List,G1["E"],label="$\mathcal{G}(4,7)$")
plt.plot(List,G2["E"],label="$\mathcal{G}(5,6)$")
plt.legend()
plt.ylabel("<E>")
plt.xlabel("Iteration")
plt.savefig('h4p3bEcompare.png')
plt.show()


# In[12]:


plt.plot(List,G1["cov"],label="$\mathcal{G}(4,7)$")
plt.plot(List,G2["cov"],label="$\mathcal{G}(5,6)$")
plt.legend()
plt.ylabel("COV")
plt.xlabel("Iteration")
plt.savefig('h4p3bCOVcompare.png')
plt.show()

