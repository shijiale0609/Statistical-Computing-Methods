#Jiale Shi
#Statistic Computing
#Final HW 
#Problem 3


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# For consistency, please use seed(21) in your reports in all computer languages.
np.random.seed(398042)

# This function defines the true path of the drone for N time steps.
def data_gen(N):
    def R3(theta):
        return [[np.cos(theta), -np.sin(theta), 0.],
                [np.sin(theta),  np.cos(theta), 0.],
                [0.,       0.,       1.]]

    def R2(theta):
        return [[np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]]

    def R3i(theta):
        return np.linalg.inv(R3(theta))

    J2 = np.eye(2)

    def J1(alphl, betl, alphr, betr, l):
        return [[np.sin(alphl + betl),   -np.cos(alphl + betl),   -l * np.cos(betl)],
                [np.sin(alphr + betr),   -np.cos(alphr + betr),   -l * np.cos(betr)]]

    j1 = J1(np.pi/2, 0, -np.pi/2, np.pi, 1)

    j1i = np.linalg.pinv(j1)

    x = 0
    y = 0
    theta = 0

    vl = np.random.randn() * 0.05 + 0.05
    vr = np.random.randn() * 0.05 + 0.05

    data = np.zeros((N, 5))

    for i in range(N):
        if (i+1) % 10 == 0:
            vl = vl + np.random.rand() * 0.2 - 0.05
            vr = vr + np.random.rand() * 0.2 - 0.05

        vl = max(-0.1, min(0.3, vl))
        vr = max(-0.1, min(0.3, vr))

        r = np.matmul(np.matmul(np.matmul(R3i(theta), j1i), J2), [vl, vr])
        #print r
        x = x + r[0]
        y = y - r[1]
        theta = theta + r[2]

        data[i, 0] = x
        data[i, 1] = y
        data[i, 2] = theta
        data[i, 3] = vl
        data[i, 4] = vr

    return data

N = 2500
T = data_gen(N)
plt.plot(T[:,0],T[:,1],label="True Path")
plt.legend()
plt.show()


#Get noisy GPS data G= T+N(0,4)
Gx = np.array([0.0]*2500)
Gy = np.array([0.0]*2500)

np.random.seed(333)
for i in range(0,N):
    r = scipy.stats.norm.rvs(loc=0,scale=np.sqrt(4));
    theta = np.random.rand(1)*2*np.pi
    Gx[i] = r*np.cos(theta)+T[i,0];
    Gy[i] = r*np.sin(theta)+T[i,1];

plt.plot(T[300:600,0],T[300:600,1],label="true path")    
plt.plot(Gx[300:600],Gy[300:600],label="noisy GPS")
plt.legend()
plt.show()


# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf


Q = 1e-5 # process variance
sz = (N,)
# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
xP=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
xPminus=np.zeros(sz)    # a priori error estimate
xK=np.zeros(sz)         # gain or blending factor
yhat=np.zeros(sz)      # a posteri estimate of y
yP=np.zeros(sz)         # a posteri error estimate
yhatminus=np.zeros(sz) # a priori estimate of y
yPminus=np.zeros(sz)    # a priori error estimate
yK=np.zeros(sz)         # gain or blending factor

R = 0.02**2 # estimate of measurement variance, change to see effect

#intial guesses
xhat[0] = Gx[0]
xP[0] = 1.0
yhat[0] = Gy[0]
yP[0] = 1.0


for k in range(1,N):
    # time update for y
    yhatminus[k] = yhat[k-1]
    yPminus[k] = yP[k-1]+Q

    # measurement update for y
    yK[k] = yPminus[k]/( yPminus[k]+R )
    yhat[k] = yhatminus[k]+yK[k]*(Gy[k]-yhatminus[k])
    yP[k] = (1-yK[k])*yPminus[k]
    
    # time update for x
    xhatminus[k] = xhat[k-1]
    xPminus[k] = xP[k-1]+Q

    # measurement update for x
    xK[k] = xPminus[k]/( xPminus[k]+R )
    xhat[k] = xhatminus[k]+xK[k]*(Gx[k]-xhatminus[k])
    xP[k] = (1-xK[k])*xPminus[k]
    
#plot for 300-600 steps    
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

xmajorLocator = MultipleLocator(10)
xmajorFormatter = FormatStrFormatter("%10d")
xminorLocator = MultipleLocator(5)
 
ymajorLocator = MultipleLocator(10)
ymajorFormatter = FormatStrFormatter("%10d")
yminorLocator = MultipleLocator(5)

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(T[300:600,0],T[300:600,1],c='r',linewidth=2,label="True path")  
ax.plot(Gx[300:600],Gy[300:600],c='orange',label="GPS Observed")
ax.plot(xhat[300:600],yhat[300:600],c='b',linewidth=2,label="Kalman Filter")
ax.legend(fontsize='xx-large',loc='upper left')
plt.xlim(10,90)
plt.ylim(70,130)
plt.xlabel("$x$",size=30)
plt.ylabel("$y$",size=30)

for side in ax.spines.keys():
     ax.spines[side].set_linewidth(3)
        
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
#ax.xaxis.set_minor_locator(xminorLocator)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
#ax.yaxis.set_minor_locator(yminorLocator)

ax.tick_params(which = 'both',direction='in',width=1,colors='black',#grid_color='r',grid_alpha=0.5,
               bottom = True,top=True,left=True,right=True)
ax.tick_params(which = 'major',direction='in',length=10,labelsize=18)
ax.tick_params(which = 'minor',direction='in',length=5)
plt.savefig("HW6P3_300steps.png",dpi=400)
plt.show()

#plot for 2500 steps
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

xmajorLocator = MultipleLocator(100)
xmajorFormatter = FormatStrFormatter("%10d")
xminorLocator = MultipleLocator(50)
 
ymajorLocator = MultipleLocator(100)
ymajorFormatter = FormatStrFormatter("%10d")
yminorLocator = MultipleLocator(50)

fig,ax = plt.subplots(figsize=(8,6))
ax.plot(T[:,0],T[:,1],c='r',linewidth=1,label="True path")  
ax.plot(Gx,Gy,c='orange',label="GPS Observed")
ax.plot(xhat,yhat,c='b',linewidth=1,label="Kalman Filter")
ax.legend(fontsize='xx-large',loc='lower right')
plt.xlim(0,400)
plt.ylim(0,500)
plt.xlabel("$x$",size=30)
plt.ylabel("$y$",size=30)

for side in ax.spines.keys():
     ax.spines[side].set_linewidth(3)
        
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)
#ax.xaxis.set_minor_locator(xminorLocator)

ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_major_formatter(ymajorFormatter)
#ax.yaxis.set_minor_locator(yminorLocator)

ax.tick_params(which = 'both',direction='in',width=1,colors='black',#grid_color='r',grid_alpha=0.5,
               bottom = True,top=True,left=True,right=True)
ax.tick_params(which = 'major',direction='in',length=10,labelsize=18)
ax.tick_params(which = 'minor',direction='in',length=5)
plt.savefig("HW6P3_2500steps.png",dpi=400)
plt.show()
