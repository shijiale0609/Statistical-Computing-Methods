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
x = np.linspace(1,41,40)
A = np.random.exponential(scale=5,size=20)
##############################################################
# INSERT CODE BELOW
##############################################################

lambda_MLE = 
##############################################################
# INSERT CODE ABOVE
##############################################################

MAP = []
MAP_est= []
for alpha in range (1,41):
    beta = 100
    n = len(A)
##############################################################
# INSERT CODE BELOW
##############################################################
    lambda_MAP = 
##############################################################
# INSERT CODE ABOVE
##############################################################
    mse = ((0.2 - lambda_MAP) ** 2).mean(axis=None)
    MAP.append(mse)
MAP_lambda = np.array(MAP)
plt.xlabel(r'$\alpha$', fontsize = 16)
plt.ylabel('MSE', fontsize = 16)
plt.plot(x,MAP_lambda, marker='o', ms = 10, color='k')
plt.savefig('Solution-6B.png')
plt.show()


