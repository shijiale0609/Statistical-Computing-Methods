'''
Statistical Computing for Scientists and Engineers
Homework 2
Fall 2018
University of Notre Dame
'''
import numpy as np
import matplotlib.pyplot as plt

#-- Part (a)
##############################################################
# INSERT CODE BELOW
##############################################################
# define the function

# define the exact solution

# do MC integration

# report exact value and MC integration solution


#-- Part (b)


# Use this to plot errors. Use L-2 norm for error.
# plt.plot(  ,  , alpha=0.7)
plt.xlabel("N")
plt.ylabel("Error")
plt.show()

#-- Part (c)


# use this to plot histograms. Plot 4 histograms.
plt.hist(MC_int, bins=30)
plt.xlabel("MC solution")
plt.show()

##############################################################
# INSERT CODE ABOVE
##############################################################