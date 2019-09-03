'''
Statistical Computing for Scientists and Engineers
Homework 2
Fall 2018
University of Notre Dame
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
import pandas as pd 
data = pd.read_csv('data.csv',header=None)
data = np.array(data)
y = data[:,0]
n = data[:,1]
print (y)
N = len(n)
#log-likelihood 
def log_likelihood(alpha,beta,y,n):
    '''
    marginal posterior distribution-log_likelihood part
    @args: y = training data, parameters: alpha, beta and dataset:n
    @returns: log_likelihood
    '''
    log_likelihood = 0
    for Y,N in zip(y,n):
	##############################################################
	# INSERT CODE BELOW
	##############################################################	
        log_likelihood+= 
	##############################################################
	# INSERT CODE ABOVE
	##############################################################

    return log_likelihood

def log_prior(A,B):
    '''
    marginal posterior distribution-log_prior part
    @args: parameters: alpha, beta
    @returns: prior
    '''
    ##############################################################
    # INSERT CODE BELOW
    ##############################################################	
    log_prior = 
    ##############################################################
    # INSERT CODE ABOVE
    ##############################################################
    return log_prior

def to_beta(x,y):

    return np.exp(y)/(np.exp(x)+1)

def to_alpha(x,y):

    return np.exp(x)*trans_to_beta(x,y)


X,Y = np.meshgrid(np.arange(-2.3,-1.3,0.01),np.arange(1,5,0.01))
param_space = np.c_[X.ravel(), Y.ravel()]
df= pd.DataFrame(param_space, columns=['X','Y'])


df['alpha']= to_alpha(df.X,df.Y)
df['beta'] = to_beta(df.X,df.Y)

df['log_posterior'] = log_prior(df.alpha,df.beta) + log_likelihood(df.alpha,df.beta, y,n)
df['log_jacobian'] = np.log(df.alpha) + np.log(df.beta)

df['transformed'] = df.log_posterior+df.log_jacobian
df['exp_trans'] = np.exp(df.transformed - df.transformed.max())

surface = df.set_index(['X','Y']).exp_trans.unstack().values.T

fig, ax = plt.subplots(figsize = (8,8),facecolor='white')
ax.contourf(X,Y, surface)
ax.set_xlabel(r'$\log(\alpha/\beta)$', fontsize = 16)
ax.set_ylabel(r'$\log(\alpha+\beta)$', fontsize = 16)
ix_z,ix_x = np.unravel_index(np.argmax(surface, axis=None), surface.shape)
print ('x:co-ordinate_value:', np.round(X[0,ix_x],2))
print ('y:co-ordinate_value:', np.round(Y[ix_z,0],2))
ax.scatter([X[0,ix_x]], [Y[ix_z,0]], color = 'red')

text= r"$({a},{b})$".format(a = np.round(X[0,ix_x],2), b = np.round(Y[ix_z,0],2))

ax.annotate(text,xy = (X[0,ix_x],Y[ix_z,0]), xytext=(-1.6, 3.5),
            ha = 'center',
            fontsize = 16,
            color = 'white',
            arrowprops=dict(facecolor='black',));
plt.savefig('HW2-P1b.png', facecolor=fig.get_facecolor(), transparent=True)
plt.show()




