
# coding: utf-8

# In[15]:


'''
Statistical Computing for Scientists and Engineers
Homework 2
Fall 2018
University of Notre Dame
'''
import numpy as np
import matplotlib.pyplot as plt

#exact solution
I = 0.25*(pow(4,4)-pow(3,4))+(5*4*np.sin(4)+5*np.cos(4))-(5*3*np.sin(3)+5*np.cos(3))
print (I)

#-- Part (a)
xa= np.random.uniform(3,4,10000)
fa = pow(xa,3)+5*xa*np.cos(xa)
sol_a = np.mean(fa)
print (sol_a)
plt.scatter(xa,fa)
plt.title("N=10,000 samples")
plt.xlabel("x")
plt.ylabel("f=x^3+5*x*cos(x)")
plt.show()


# In[38]:


#-- Part (b)

Nb = np.array([0]*991)
Ib = np.array([0.0]*991)
Errb = np.array([0.00]*991)

for i in range(0,991):
    Nb[i] = i+10
for i in range(0,991):
    xb= np.random.uniform(3,4,Nb[i])
    Ib[i] = np.mean(pow(xb,3)+5*xb*np.cos(xb))
    Errb[i] = (Ib[i]-I)/I


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(Nb, Ib , alpha=0.7)
plt.xlabel("N")
plt.ylabel("I")

plt.subplot(1,2,2)
plt.plot(Nb, Errb , alpha=0.7)
plt.xlabel("N")
plt.ylabel("Error")
plt.show()


# In[40]:


#-- Part (c)


# use this to plot histograms. Plot 4 histograms.
MCc1 = np.array([0.0]*10000)
MCc2 = np.array([0.0]*10000)
MCc3 = np.array([0.0]*10000)
MCc4 = np.array([0.0]*10000)

for i in range(0,10000):
    xc1= np.random.uniform(3,4,100)
    MCc1[i] = np.mean(pow(xc1,3)+5*xc1*np.cos(xc1))
    xc2= np.random.uniform(3,4,1000)
    MCc2[i] = np.mean(pow(xc2,3)+5*xc2*np.cos(xc2))
    xc3= np.random.uniform(3,4,10000)
    MCc3[i] = np.mean(pow(xc3,3)+5*xc3*np.cos(xc3))
    xc4= np.random.uniform(3,4,100000)
    MCc4[i] = np.mean(pow(xc4,3)+5*xc4*np.cos(xc4))
    print (i)



# In[43]:


plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.title("N=100")
plt.hist(MCc1, bins=30)
plt.xlabel("MC solution")

plt.subplot(2,2,2)
plt.title("N=1000")
plt.hist(MCc2, bins=30)
plt.xlabel("MC solution")


plt.subplot(2,2,3)
plt.title("N=10000")
plt.hist(MCc3, bins=30)
plt.xlabel("MC solution")

plt.subplot(2,2,4)
plt.title("N=100000")
plt.hist(MCc4, bins=30)
plt.xlabel("MC solution")


plt.show()



##############################################################
# INSERT CODE ABOVE
##############################################################



