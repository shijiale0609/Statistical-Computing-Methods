'''
Nicholas Geneva
ngeneva@nd.edu
October 31, 2017
'''
import sys
import random
import numpy as np
import scipy as sc
import matplotlib as mlp
import matplotlib.pyplot as plt

from matplotlib import rc
from scipy import special
from scipy import stats

def monotoneSIS(nx, ny, p=0.5, atmpt=1):
    '''
    Conducts a Monotone (move up or right) self-avoiding chain random walk and 
    accepts if it ends in the top right corner (nx,ny)
    Args:
        nx (int) = number of cells in x deminsion
        ny (int) = number of cells in y deminsion
        p (float) = probability of choosing right
        atmp (int) = number of attempts to creat valid chain (NOT SET BY USER)
    Returns:
        pos (nd.array) = [nsteps, 2] array of positions the chain walked
        atmpt (int) = number of attempts made to create valid chain
    '''
    pos = [[0,0]]
    prob = []
    cpos_dir = np.array([[0,1],[1,0]]) #Up, or right

    for i in range(nx+ny):
        cpos = []
        #First check for neighbors, only choose from valid locations
        for j, dir0 in enumerate(cpos_dir):
            x = pos[-1][0]+dir0[0]
            y = pos[-1][1]+dir0[1]
            if((not [x,y] in pos) and x >= 0 and x <= nx and y >= 0 and y <= ny):
                cpos.append([x,y])

        #Choose next location (right or up)
        pos.append(cpos[np.random.randint(len(cpos))])
        prob.append(1.0/len(cpos))

        #Chain landed in destination to accept it
        if(pos[-1] == [nx,ny]):
            return np.array(pos), np.array(prob), atmpt
        #Failed chain
        if(pos[-1][0] > nx or pos[-1][1] > ny):
            break

    #Chain left the domain, so lets restart (Dangerous but yolo)
    return monotoneSIS(nx, ny, p, atmpt+1)

def selfAvoidSIS(nx, ny, atmpt=1):
    '''
    Conducts a self-avoiding chain random walk within the [nx ny] bloack
    and accepts if it ends in the top right corner (nx,ny)
    Args:
        nx (int) = number of cells in x deminsion
        ny (int) = number of cells in y deminsion
        atmp (int) = number of attempts to creat valid chain (NOT SET BY USER)
    Returns:
        pos (nd.array) = [nsteps, 2] array of positions the chain walked
        prob (nd.array) = [nsteps, 1] array of positions the chain walked
        atmpt (int) = number of attempts made to create valid chain
    '''
    pos = [[0,0]]
    prob = []
    cpos_dir = np.array([[0,1],[0,-1],[1,0],[-1,0]])

    while(True): #Go Go power rangers
        cpos = []
        #First check for neighbors
        for j, dir0 in enumerate(cpos_dir):
            x = pos[-1][0]+dir0[0]
            y = pos[-1][1]+dir0[1]
            if((not [x,y] in pos) and x >= 0 and x <= nx and y >= 0 and y <= ny):
                cpos.append([x,y])

        #Chain has trapped itself so kill it
        if(len(cpos) == 0):
            break

        pos.append(cpos[np.random.randint(len(cpos))])
        prob.append(1.0/len(cpos))
        #Chain landed in destination to accept it
        if(pos[-1] == [nx,ny]):
            return np.array(pos), np.array(prob), atmpt

    #Failed chain, so lets restart (Dangerous but yolo)
    return selfAvoidSIS(nx, ny, atmpt+1)

if __name__ == '__main__':
    plt.close('all')
    mlp.rcParams['font.family'] = ['times new roman'] # default is sans-serif
    rc('text', usetex=True)

    #========== Problem 2 (a) ==========
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 5 Problem 2(a)', fontsize=14)

    N = 500 #number of path samples
    pos = [];  prob = []; atmps = np.zeros(N)
    for i in range(N):
        pos0, prob0, atmps[i] = monotoneSIS(10,10,0.5)
        pos.append(pos0)
        prob.append(prob0)
        ax.plot(pos[-1][:,0],pos[-1][:,1])

    #Total number of paths zn, note that the probability of one polymer is 1/zn or (1/2)**(2n)
    alpha = float(N)/np.sum(atmps) #Acceptance rate of polymers
    avg_num = sum([1.0/np.prod(prob0) for prob0 in prob])/float(N)
    print('=Monotone Polymer Chains=')
    print('Polymer acceptance rate %0.2f' % (alpha))
    print('Number of polymer chains in box %0.2f' % (avg_num))
    print('Exacty number is 184756.')
    print('Relative Error %0.2f%%' % (100*np.abs(184756 - avg_num)/184756.0))

    #Heat map
    x = np.arange(0,10)
    y = np.arange(0,10)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    #Get the number of polymer nodes at each grid point
    for (i,j), x0 in np.ndenumerate(X):
        y0 = Y[i,j] 
        for k, pos0 in enumerate(pos):
             Z[i,j] += ((pos0[:,0] == x0)*(pos0[:,1] == y0)).sum()

    cmap = plt.cm.hot
    levels = 30
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 5 Problem 2(a) Heatmap', fontsize=14)
    cs = ax.contourf(X, Y, Z, levels, cmap=plt.cm.get_cmap(cmap, levels), linewidth=0.5, zorder=1)
    f.colorbar(cs, ax=ax)


    #========== Problem 2 (b) ==========
    #Set up subplots
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 5 Problem 2(b)', fontsize=14)

    N = 500 #number of path samples
    pos = []; prob = []; atmps = np.zeros(N)
    minX = 0; minY = 0; maxX = 0; maxY = 0
    for i in range(N):
        pos0, prob0, atmps[i] = selfAvoidSIS(10,10)
        pos.append(pos0)
        prob.append(prob0)
        ax.plot(pos[-1][:,0],pos[-1][:,1])

        #Check for global mins/maxes
        if(max(pos[-1][:,0]) > maxX):
            maxX = max(pos[-1][:,0])
        if(min(pos[-1][:,0]) < minX):
            minX = min(pos[-1][:,0])
        if(max(pos[-1][:,1]) > maxY):
            maxY = max(pos[-1][:,1])
        if(min(pos[-1][:,1]) < minY):
            minY = min(pos[-1][:,1])

    alpha = float(N)/np.sum(atmps) #Acceptance rate of polymers
    avg_len = sum([pos0.shape[0] for pos0 in pos])/float(N)
    avg_num = sum([1.0/np.prod(prob0) for prob0 in prob])/float(N)
    print('=Self Avoiding Polymer Chains=')
    print('Polymer acceptance rate %0.2f' % (alpha))
    print('Polymer average length %0.2f' % (avg_len))
    print('Polymer average number %0.4E' % (avg_num))

    #Heat map
    x = np.arange(minX,maxX+1)
    y = np.arange(minY,maxY+1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    #Get the number of polymer nodes at each grid point
    for (i,j), x0 in np.ndenumerate(X):
        y0 = Y[i,j] 
        for k, pos0 in enumerate(pos):
             Z[i,j] += ((pos0[:,0] == x0)*(pos0[:,1] == y0)).sum()

    cmap = plt.cm.hot
    levels = 30
    f, ax = plt.subplots(1, 1, figsize=(7, 6))
    f.suptitle('Homework 5 Problem 2(b) Heatmap', fontsize=14)
    cs = ax.contourf(X, Y, Z, levels, cmap=plt.cm.get_cmap(cmap, levels), linewidth=0.5, zorder=1)
    f.colorbar(cs, ax=ax)

    plt.show()

    

