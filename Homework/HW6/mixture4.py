#Jiale Shi
#Transfer EM algorithm matlab to python
import numpy as np
import numpy.random as random

def mixtures4(y, kmin, kmax, regularize,th, covoption):
    verb = 1;
    bins = 40;
    dl = [];
    [dimens,npoints] = size(y)
    switch covoption:
           case 0:
                npars = (dimens+dimens*(dimens+1)/2);
           case 1:
                npars = 2*dimens;
           case 2:
                npars = dimens;
           case 3:
                npars = dimens;
           otherwise:
                # the default is to assume free covariances
                npars = (dimens+dimens*(dimens+1)/2);

    nparsovers = npars/2;
    axis1 = 1;
    axis2 = 2;

    k = kmax;
    indic = np.zeros(k,npoints);

    randindex = randperm(npoints);
    randindex = randindex(1:k)
    estmu = y(: randindex)
    
