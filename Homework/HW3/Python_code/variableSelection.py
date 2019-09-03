import random
import numpy as np
import scipy as sc
import util
import log

from scipy import special
from scipy import stats

def variableSelection(x0, y0, beta_tilde, c, clim):
    '''
    Performs variable (model) selection by computing the evidence of each model
    @params:
        x0 (np.array): array of input data
        y0 (np.array): array of training data
        beta_tilde (np.array): prior hyperparameter
        c (float): prior hyperparameter
        clim (int): upper limit of the c summation for the non-informative prior
    '''
    (N,K) = x0.shape
    x = np.zeros((N,K+1)) + 1
    x[:,1:] = x0 #Expanding with a colomn of ones for bias terms

    zeller_posterior = np.zeros((2**K,2))
    post_informative = np.zeros((2**K,2))
    post_noninformative = np.zeros((2**K,2))
    #Save the index of each model in the array for when we max sort it
    post_informative[:,0] = range(2**K)
    post_noninformative[:,0] = range(2**K)

    log.log("Starting model variable selection for both priors")

    for i in range(2**K):
        (gamma_index, q) = util.getGammaIndexes(K, i)
        
        x_g = x[:,gamma_index]
        bt_g = beta_tilde[gamma_index]

        xtxi =  np.linalg.inv(x_g.T.dot(x_g))
        c0 = y0.T.dot(x_g.dot(xtxi).dot(x_g.T)).dot(y0)
        c1 = y0.T.dot(y0)-c/(c+1.0)*c0+1/(c+1.0)*bt_g.T.dot(x_g.T.dot(x_g)).dot(bt_g) -\
            2.0/(c+1.0)*y0.T.dot(x_g).dot(bt_g)        
        
        zeller_posterior[i,0] = (c+1.0)**(-0.5*(q+1))*c1**(-0.5*N)

        zeller_noninform = 0
        for ci in range(1,int(clim)):
            c1 = (1.0/ci)*(ci+1)**(-0.5*(q+1))
            zeller_noninform += c1*(y0.T.dot(y0) - ci/(ci+1.0)*c0)**(-0.5*N)

        zeller_posterior[i,1] = zeller_noninform
    
    log.sucess("Variable selection posteriors calculated")
    log.log("Normalizing and sorting Model Evidence")

    #Normalize model evidence
    post_informative[:,1] = zeller_posterior[:,0]/np.sum(zeller_posterior[:,0])
    post_noninformative[:,1] = zeller_posterior[:,1]/np.sum(zeller_posterior[:,1])
    
    #Sort array largest to smallest based on model evidence
    #https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    post_informative_sort = post_informative[post_informative[:,1].argsort()[::-1]]
    post_noninformative_sort = post_noninformative[post_noninformative[:,1].argsort()[::-1]]

    return post_informative_sort, post_noninformative_sort

def gibbsSamplingInformative(x0, y0, beta_tilde, c, T, T0):
    '''
    Performs Gibb's sampling for Zelner's Informative prior
    @params:
        x0 (np.array): array of input data
        y0 (np.array): array of training data
        beta_tilde (np.array): prior hyperparameter
        T (int): numder of samples to take
        T0 (int): "burn in" samples to ignore
    '''
    T = int(T); T0 = int(T0)
    (N,K) = x0.shape
    x = np.zeros((N,K+1)) + 1
    x[:,1:] = x0 #Expanding with a colomn of ones for bias terms

    zeller_evid = np.zeros(2)
    gammas = np.zeros(T)
    post_informative = np.zeros((2**K, 2))
    beta_evidence = np.zeros((K))
    post_informative[:,0] = range(2**K)

    gam_index = random.randint(0, 2**K)
    (t_gamma0, q) = util.getGammaIndexes(K, int(gam_index))

    log.log("Staring Zeller Informative Prior Gibb's sampling")
    log.warning("This will take a while...")

    for i in range(T):
        
        if(i%100 == 0):
            log.print_progress(i,T)

        for j in range(K):
            gam_i0 = gam_index; gam_i1 = gam_index
            if(int(gam_index/2**j)%2 == 0):
                gam_i1 = gam_index+2**j #With current model parameter
            else:
                gam_i0 = gam_index-2**j #With out current model parameter

            
            for i0, gam0 in enumerate([int(gam_i0), int(gam_i1)]):        
                (t_gamma, q) = util.getGammaIndexes(K, gam0)
                x_g = x[:,t_gamma]
                bt_g = beta_tilde[t_gamma]

                xtxi =  np.linalg.inv(x_g.T.dot(x_g))
                c0 = y0.T.dot(x_g.dot(xtxi).dot(x_g.T)).dot(y0)
                c1 = y0.T.dot(y0)-c/(c+1.0)*c0+1/(c+1.0)*bt_g.T.dot(x_g.T.dot(x_g)).dot(bt_g) -\
                    2.0/(c+1.0)*y0.T.dot(x_g).dot(bt_g)        
            
                zeller_evid[i0] = (c+1.0)**(-0.5*(q+1))*c1**(-0.5*N)
            
            zeller_evid_norm = zeller_evid[0]/np.sum(zeller_evid, 0)
            if(random.random() < zeller_evid_norm):
                gam_index = gam_i0
            else:
                gam_index = gam_i1
        gammas[i] = gam_index

    log.print_progress(T,T)
    log.sucess("Gibb's Sampling for informative prior complete!")
    log.log("Calculating Model Evidence")

    #Count all instances of a certain model to get model evidence
    for i in range(2**K):
        count = np.count_nonzero(gammas[T0:T] == i)
        post_informative[i,1] = count / (T - T0 + 1.0)

    post_informative_sort = post_informative[post_informative[:,1].argsort()[::-1]]

    #Count all instances of a certain certain to get variable evidence
    for i in range(K):
        count = np.where((gammas[T0:T]/(2**i)).astype(int)%2 == 1)[0].shape[0]
        beta_evidence[i] = count / (T - T0 + 1.0)

    return post_informative_sort, beta_evidence

def gibbsSamplingNonInformative(x0, y0, beta_tilde, clim, T, T0):
    '''
    Performs Gibb's sampling for Zelner's Non-Informative prior
    @params:
        x0 (np.array): array of input data
        y0 (np.array): array of training data
        beta_tilde (np.array): prior hyperparameter
        clim (int): upper limit of the c summation for the non-informative prior
        T (int): numder of samples to take
        T0 (int): "burn in" samples to ignore
    '''
    T = int(T); T0 = int(T0)
    (N,K) = x0.shape
    x = np.zeros((N,K+1)) + 1
    x[:,1:] = x0 #Expanding with a colomn of ones for bias terms

    zeller_evid = np.zeros(2)
    gammas = np.zeros(T)
    post_noninfo = np.zeros((2**K, 2))
    beta_evidence = np.zeros((K))
    post_noninfo[:,0] = range(2**K)

    gam_index = random.randint(0, 2**K)
    (t_gamma0, q) = util.getGammaIndexes(K, int(gam_index))

    log.log("Staring Zeller Non-informative Prior Gibb's sampling")
    log.warning("This will take a while...")
    for i in range(T):

        if(i%100 == 0):
            log.print_progress(i,T)

        for j in range(K):
            gam_i0 = gam_index; gam_i1 = gam_index
            if(int(gam_index/2**j)%2 == 0):
                gam_i1 = gam_index+2**j #With current model parameter
            else:
                gam_i0 = gam_index-2**j #With out current model parameter

            
            for i0, gam0 in enumerate([int(gam_i0), int(gam_i1)]):        
                (t_gamma, q) = util.getGammaIndexes(K, gam0)
                x_g = x[:,t_gamma]
                bt_g = beta_tilde[t_gamma]

                xtxi =  np.linalg.inv(x_g.T.dot(x_g))
                c0 = y0.T.dot(x_g.dot(xtxi).dot(x_g.T)).dot(y0)
                #Marginalize over c
                zeller_noninform = 0
                for ci in range(1,int(clim)):
                    c1 = (1.0/ci)*(ci+1)**(-0.5*(q+1))
                    zeller_noninform += c1*(y0.T.dot(y0) - ci/(ci+1.0)*c0)**(-0.5*N)

                zeller_evid[i0] = zeller_noninform
            
            #Find variable evidence, and determine if to keep it or not
            zeller_evid_norm = zeller_evid[0]/np.sum(zeller_evid, 0)
            if(random.random() < zeller_evid_norm): #Bernoulli
                gam_index = gam_i0
            else:
                gam_index = gam_i1
        gammas[i] = gam_index

    log.print_progress(T,T)
    log.sucess("Gibb's Sampling for non-informative prior complete!")
    log.log("Calculating Model Evidence")

    #Count all instances of a certain model to get model evidence
    for i in range(2**K):
        count = np.count_nonzero(gammas[T0:T] == i)
        post_noninfo[i,1] = count / (T - T0 + 1.0)

    post_noninfo_sort =  post_noninfo[ post_noninfo[:,1].argsort()[::-1]]

    #Count all instances of a certain certain to get variable evidence
    for i in range(K):
        count = np.where((gammas[T0:T]/(2**i)).astype(int)%2 == 1)[0].shape[0]
        beta_evidence[i] = count / (T - T0 + 1.0)

    return  post_noninfo_sort, beta_evidence