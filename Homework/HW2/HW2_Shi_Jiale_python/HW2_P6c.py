
# coding: utf-8

# In[12]:


'''
Statistical Computing for Scientists and Engineers
Homework 2
Fall 2018
University of Notre Dame
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from sklearn.metrics import mean_squared_error
x = np.linspace(1,501,500)
print (x.shape)
main_MLE = []
main_MAP = []
for i in range (1,501):
	A = np.random.exponential(scale=5,size=i)
	##############################################################
	# INSERT CODE BELOW
	##############################################################

	lambda_MLE = 1/np.mean(A)
	##############################################################
	# INSERT CODE ABOVE
	##############################################################
	mse_MLE  = ((0.2 - lambda_MLE) ** 2).mean(axis=None)
	main_MLE.append(mse_MLE)
	alpha = 30
	beta = 100
	n = len(A)
	##############################################################
	# INSERT CODE BELOW
	##############################################################
	lambda_MAP = (n+alpha-1)/(np.sum(A)+beta)
	##############################################################
	# INSERT CODE ABOVE
	##############################################################
	mse_MAP = ((0.2 - lambda_MAP) ** 2).mean(axis=None)
	main_MAP.append(mse_MAP)
#print (mean)
#print (map_E)
main_MLE_value = np.array(main_MLE)
main_MAP_value = np.array(main_MAP)
print (main_MLE_value.shape)
plt.plot(x,main_MLE_value)
plt.plot(x,main_MAP_value)
plt.legend(['MLE','MAP'])
plt.xlabel('N', fontsize = 16)
plt.ylabel('MSE', fontsize = 16)
plt.savefig('Solution-6C.png')
plt.show()

