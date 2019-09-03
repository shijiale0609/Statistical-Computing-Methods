
# coding: utf-8

# In[1]:


import random
import numpy as np
import scipy.stats
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.animation as animation



x1 = 20;y1=0;
x2 = 20*np.cos(np.pi*2/5);y2 = 20*np.sin(np.pi*2/5);
x3 = 20*np.cos(np.pi*4/5);y3 = 20*np.sin(np.pi*4/5);
x4 = 20*np.cos(np.pi*6/5);y4 = 20*np.sin(np.pi*6/5);
x5 = 20*np.cos(np.pi*8/5);y5 = 20*np.sin(np.pi*8/5);

ak = np.array([0.0]*101)
for i in range(0,101):
    ak[i]=0.01*i;
    
def target1D(x):
    #return scipy.stats.norm.pdf(x, loc=20,scale=1) + scipy.stats.norm.pdf(x, loc= -20,scale=1)
    return np.exp(-pow(x-20,2)/2)+np.exp(-pow(x+20,2)/2)
    
def proposal1D(x,x0):
    return scipy.stats.norm.pdf(x, loc=x0, scale=np.sqrt(6))

def target2D(x,y):
    cov = [[1.0, 0.0], [0.0, 1.0]]
    #return np.exp(-pow(x-x1,2)/2-pow(y-y1,2)/2) \
       #     + np.exp(-pow(x-x2,2)/2-pow(y-y2,2)/2) \
        #    + np.exp(-pow(x-x3,2)/2-pow(y-y3,2)/2) \
         #   + np.exp(-pow(x-x4,2)/2-pow(y-y4,2)/2) \
          #  + np.exp(-pow(x-x5,2)/2-pow(y-y5,2)/2)
    return  scipy.stats.multivariate_normal.pdf([x, y], mean=[x1, y1],    cov=cov)             + scipy.stats.multivariate_normal.pdf([x, y], mean=[x2, y2], cov=cov)             + scipy.stats.multivariate_normal.pdf([x, y], mean=[x3, y3], cov=cov)            + scipy.stats.multivariate_normal.pdf([x, y], mean=[x4, y4], cov=cov)             + scipy.stats.multivariate_normal.pdf([x, y], mean=[x5, y5], cov=cov) 
            
            
def proposal2D(x,y,x0,y0):
    #return  np.exp(-pow(x-x0,2)/(2*6)-pow(y-y0,2)/(2*6))
    mean = [x0, y0]
    cov = [[np.sqrt(6), 0.0], [0.0, np.sqrt(6)]]
    rv = scipy.stats.multivariate_normal(mean=mean,cov=cov)
    return  rv.pdf([x,y])

Num=3000;
weight = np.array([1/Num]*Num)

x = np.array([0.0]*Num)
y = np.array([0.0]*Num)

plt.figure(figsize=(8,8))
#plt.hist(x,bins=50, alpha=0.4, label=u'sampled histogram', normed=True) 
#plt.plot(x,y,'r', marker='o',linewidth=0,label=u'True distribution') # f(x) is the True distribution
plt.scatter(x,y,label="K=0")
plt.legend()
plt.xlim([-22,22])
plt.ylim([-22,22])
#plt.savefig('h4p1b4.png')
plt.show()


fig = plt.figure(figsize=(10,10))

ims = []
for k in range(1,101):
    for i in range(0,Num):
           weight[i] = weight[i] * pow(target2D(x[i],y[i]),ak[k])/pow(target2D(x[i],y[i]),ak[k-1])
                  
    weight = weight/sum(weight)
    ESS = 1/np.sum(pow(weight,2))   
    print (ESS)
    if (ESS<=Num/2):
         weight = np.array([1/Num]*Num)
                  
    
    for i in range(0,Num):
        q = scipy.stats.multivariate_normal([0,0],[[np.sqrt(6),0.0],[0.0,np.sqrt(6)]])
        r = q.rvs()
        xn = r[0]+x[i];
        yn = r[1]+y[i];
        a = min(1,target2D(xn,yn)/target2D(x[i],y[i])*proposal2D(x[i],y[i],xn,yn)/proposal2D(xn,yn,x[i],y[i]));
        u = np.random.rand(1);
        if (u<a):
            x[i] = xn;
            y[i] = yn;
    print(k)
    im = plt.scatter(x,y, animated=True)
    ims.append([im])


    #if (k==5):
    #    plt.figure(figsize=(8,8))
    #plt.hist(x,bins=50, alpha=0.4, label=u'sampled histogram', normed=True) 
    #plt.plot(x,y,'r', marker='o',linewidth=0,label=u'True distribution') # f(x) is the True distribution
     #   plt.scatter(x,y,label="K=50")
     #   plt.legend()
        #plt.xlim([-4,4])
        #plt.savefig('h4p1b4.png')
      #  plt.show()
            


# In[2]:


ims
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,repeat_delay=1000)



# In[3]:


#plt.show()
ani.save('dynamic_imageshw5p2copy.html')


# In[5]:


plt.figure(figsize=(8,8))
#plt.hist(x,bins=50, alpha=0.4, label=u'sampled histogram', normed=True) 
#plt.plot(x,y,'r', marker='o',linewidth=0,label=u'True distribution') # f(x) is the True distribution
plt.scatter(x,y,label="K=100")
plt.legend()
#plt.xlim([-4,4])
#plt.savefig('h4p1b4.png')
plt.show()


# In[ ]:


#1D
Num=3000;
weight = np.array([1/Num]*Num)

x = np.array([0.0]*Num)
y = np.array([0.0]*Num)
for k in range(1,101):
    for i in range(0,Num):
           weight[i] = weight[i] * pow(target1D(x[i]),ak[k])/pow(target1D(x[i]),ak[k-1])
                  
    weight = weight/sum(weight)
                  
    #if (ESS<=N/2):
    #     weight = np.array([1/Num]*Num)
                  
    
    for i in range(0,Num):
        r = scipy.stats.norm.rvs(loc=0,scale=np.sqrt(6));
        xn = r+x[i];
        #theta = np.random.rand(1)*2*np.pi
        #xn = r*np.cos(theta)+x[i];
        #yn = r*np.sin(theta)+y[i];
        a = min(1,target1D(xn)/target1D(x[i])*proposal1D(x[i],xn)/proposal1D(xn,x[i]));
        u = np.random.rand(1);
        if (u<a):
            x[i] = xn;
            #y[i] = yn;
    print(k)
            



                  

print ("bingo")       


