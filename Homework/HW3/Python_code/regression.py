import numpy as np
import scipy as sc
import util
import log

from scipy import special
from scipy import stats

def mleRegression(x_data, t_data, std):
    '''
    Completes a MLE regression for the given data
    Args:
        x_data (np.array): array of input data
        y_data (np.array): array of training data
    '''
    log.log("Calculating MLE regression")
    (N,K) = x_data.shape
    x = np.zeros((N,K+1)) + 1
    x[:,1:] = x_data #Expanding with a colomn of ones for bias terms
    #(X'*X)^(-1)*(X')*Y
    xtx_i = np.linalg.inv(x.T.dot(x))
    beta_hat = xtx_i.dot(x.T).dot(t_data)
    #Standard variance
    y_star = (t_data - x.dot(beta_hat))
    
    s2 = y_star.dot(y_star)
    sig_hat = 1.0/(N-K-1)*s2
    std_err = (np.sqrt(np.eye(K+1)*sig_hat*xtx_i)).dot(np.zeros(K+1)+1)
    
    t = (beta_hat/std_err)
    p_right = 1.0-sc.stats.t.cdf(np.abs(t),N-K-1)
    p_left = sc.stats.t.cdf(-np.abs(t),N-K-1)
    p = p_right+p_left

    log.sucess("MLE Regression complete")
    return beta_hat, std_err, t, p

def priorExpectations(x_data, t_data, beta_hat, beta_tilde, a, b, c):
    '''
    Computes the expecation and variance of the beta_hats for conjugate priors
    @params:
        x_data (np.array): array of input data
        t_data (np.array): array of training data
        beta_hat (np.array): MLE linear regression weights
        beta_tilde (np.array): prior hyperparameter
        a (float): hyper-parameter
        b (float): hyper-parameter
        c (float): hyper-parameter
    '''
    log.log("Computing exp and var for conjugate priors")
    log.log("Hyper-parameters a:%.2f b:%.2f c:%.2f"%(a, b, c))

    (N,K) = x_data.shape
    x = np.zeros((N,K+1)) + 1
    x[:,1:] = x_data #Expanding with a colomn of ones for bias terms
    
    y_star = (t_data - x.dot(beta_hat))
    s2 = y_star.dot(y_star)
    #(X'*X)^(-1)*(X')*Y
    xtx_i = np.linalg.inv(x.T.dot(x))
    M = np.eye(K+1)/c

    c1 = np.linalg.inv(np.linalg.inv(M)+np.linalg.inv(x.T.dot(x)))
    c1 = (beta_tilde - beta_hat).T.dot(c1).dot(beta_tilde - beta_hat)
    c2 = np.linalg.inv(M+x.T.dot(x))

    exp_sig2 = (2*b + s2 + c1)/(N + 2*a - 2)
    exp_beta = c2.dot(x.T.dot(x).dot(beta_hat) + M.dot(beta_tilde))
    covar_beta = (exp_sig2)*c2
    var_beta = (np.eye(K+1)*(covar_beta)).dot(np.zeros(K+1)+1) #Get the variance (diagonal) elements

    log.sucess("MLE Regression complete")
    return exp_sig2, exp_beta, var_beta

def gPriorExpectations(x_data, t_data, beta_hat, beta_tilde, c, c0):
    '''
    Computes the expecation and variance along with the bayes factor for Zellner's
    informative g-prior
    @params:
        x_data (np.array): array of input data
        t_data (np.array): array of training data
        beta_hat (np.array): MLE linear regression weights
        beta_tilde (np.array): prior hyperparameter
        c (float): hyper-parameter
        c0 (float): model comparision hyper-parameter
    '''
    log.log("Computing exp, var and bayes factor for Zellner's informative G-prior")
    log.log("Hyper-parameters c:%.2f"%(c))

    (N,K) = x_data.shape
    x = np.zeros((N,K+1)) + 1
    q = 1
    x[:,1:] = x_data #Expanding with a colomn of ones for bias terms
    
    y_star = (t_data - x.dot(beta_hat))
    s2 = y_star.dot(y_star)
    #(X'*X)^(-1)*(X')*Y
    xtx = x.T.dot(x)
    yty = t_data.T.dot(t_data)

    exp_beta = 1.0/(c+1)*(beta_tilde + c*beta_hat)
    c1 = (s2 + (beta_tilde - beta_hat).T.dot(xtx).dot(beta_tilde-beta_hat)/(c+1))
    covar_beta = (c/(c+1.0))*(c1)/(N-2)*np.linalg.inv(xtx)
    var_beta = (np.eye(K+1)*(covar_beta)).dot(np.zeros(K+1)+1)

    b_coeff = ((c0+1)**((K+1-q)/2.0))/((c+1)**((K+1)/2.0))

    b_dem = yty - c/(c+1.0)*t_data.T.dot(x).dot(np.linalg.inv(xtx)).dot(x.T).dot(t_data) + \
             1/(c+1.0)*beta_tilde.T.dot(xtx).dot(beta_tilde) - 2/(c+1.0)*t_data.T.dot(x).dot(beta_tilde)

    #Comput the bayes factors for models excluding a specific input
    b_num = np.zeros(K+1)
    for i in range(K+1):
        x0 = np.delete(x, i, 1)
        beta_tilde0 = np.delete(beta_tilde, i, 0)
        xtx = x0.T.dot(x0)

        b_num[i] = yty - c0/(c0+1.0)*t_data.T.dot(x0).dot(np.linalg.inv(xtx)).dot(x0.T).dot(t_data) + \
            1/(c0+1.0)*beta_tilde0.T.dot(xtx).dot(beta_tilde0)-2/(c0+1.0)*t_data.T.dot(x0).dot(beta_tilde0)

    b_10 = b_coeff*(b_num/b_dem)**(0.5*N)
    
    return exp_beta, var_beta, b_10

def getHPD(x_data, t_data, beta_hat, beta_tilde, alpha):
    '''
    Get the HPD region
    @params:
        x_data (np.array): array of input data
        t_data (np.array): array of training data
        beta_hat (np.array): MLE linear regression weights
        beta_tilde (np.array): prior hyperparameter
        alpha (float): HPD range = (1-alpha)
    '''
    log.log("Computing HPD region")

    (N,K) = x_data.shape
    x = np.zeros((N,K+1)) + 1
    x[:,1:] = x_data #Expanding with a colomn of ones for bias terms
    
    xtx = x.T.dot(x)
    w = (np.eye(K+1)*np.linalg.inv(xtx)).dot(np.zeros(K+1)+1)
    
    y_star = (t_data - x.dot(beta_hat))
    s2 = y_star.dot(y_star)

    f = sc.stats.t.ppf(1-(alpha/2.0), N-K-1, loc=0, scale=1)
    hpd_interval = f*np.sqrt(w*s2/(N-K-1.0))

    hpd = np.zeros((K+1,2))
    hpd[:,0] = beta_hat[:] - hpd_interval[:]
    hpd[:,1] = beta_hat[:] + hpd_interval[:]

    return hpd

def zellnerNonInfoGPrior(x_data, y, beta_hat, beta_tilde):
    '''
    Computes the expectation and variance of Zellner's non-informative Gprior
    @params:
        x_data (np.array): array of input data
        t_data (np.array): array of training data
        beta_hat (np.array): MLE linear regression weights
        beta_tilde (np.array): prior hyperparameter
    '''
    log.log("Computing exp and var for Zellner's non-informative G-prior")

    clim = 101
    (N,K) = x_data.shape
    x = np.zeros((N,K+1)) + 1
    x[:,1:] = x_data #Expanding with a colomn of ones for bias terms
    #General commonly used numbers
    xtxi = np.linalg.inv(x.T.dot(x))
    yty = y.T.dot(y)
    c1 = y.T.dot(x).dot(xtxi).dot(x.T).dot(y)
    c2 = beta_hat.T.dot(x.T.dot(x)).dot(beta_hat)
    
    y_star = (y - x.dot(beta_hat))
    s2 = y_star.dot(y_star)

    #For non-informative we must marginalize over c
    exp_num = 0; exp_dem = 0; 
    for c in range(1,clim):
        fxy0 = (1.0/c)*(c+1)**(-(K+1)/2.0)*(yty-(c*c1)/(c+1.0))**(-N/2.0)
        exp_num += c/(c+1.0)*fxy0
        exp_dem += fxy0
    exp_beta = (exp_num/exp_dem)*beta_hat

    var_num1 = 0; var_num2 = 0; 
    for c in range(1,clim):
        fxyc0 = (c+1)**(-(K+1)/2.0)*(yty-(c*c1)/(c+1.0))**(-N/2.0)
        var_num1 += fxyc0/((N-2.0)*(c+1.0))*(s2+c2/(c+1.0))    
        var_num2 += (c/(c+1.0) - (exp_num/exp_dem))**2 * fxyc0*(1.0/c)
    
    covar_beta = (var_num1/exp_dem)*(xtxi) + np.outer(beta_hat,(var_num2/exp_dem)*beta_hat.T)
    var_beta = (np.eye(K+1)*(covar_beta)).dot(np.zeros(K+1)+1)

    return exp_beta, var_beta