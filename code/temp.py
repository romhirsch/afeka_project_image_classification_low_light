"""
Created on Wed Dec 23 19:50:22 2020

@author: Yarden Sweed and Omer Aviv
"""
#%% Declaration
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def kl_mvn(m0, s0, m1, s1):
 """
 computes KL divergence from a single Gaussian mu0,std0 to
 Gaussian mu1,std1.
 Diagonal covariances are assumed. Divergence is expressed in nats.

 - accepts stacks of means, but only one S0 and S1


 KL(p,q)=log(σ2/σ1)+(σ1^2+(μ1−μ2)^2)/2σ2^2−1/2
 """

 return np.log(s1/s0)+(s0**2+(m0-m1)**2)/(2*s1**2)-1/2



 #%% generation

m0=0.2
m1=0.2
std0=np.sqrt(1.5)
std1=np.sqrt(3)
series_size=512
d=np.zeros((7,series_size))
gen_from=np.full((7,series_size),-1)
for i in range (0,7):
    step=2**i
    ber=stats.bernoulli.rvs(0.5,size=int(series_size/step))
    for j in range (0,int(series_size/step)):
        gen_from[i][j]=ber[j]
        if (ber[j]==0):
            d[i:,step*j:step*(j+1)]=np.random.normal(loc=m0,scale=std0,size=step) #1st
        else:
            d[i:,step*j:step*(j+1)]=np.random.normal(loc=m1,scale=std1,size=step) #2nd

#%% likelihood
est_gen_from=np.full((7,series_size),-1)
for i in range (0,7):
    err=0
    step=2**i
    for j in range (0,int(series_size/step)):
        G0_like=np.log(stats.norm(m0, std0).pdf(d[i][step*j:step*(j+1)])).sum() # sumog likelihood for each RV in a series
        G1_like=np.log(stats.norm(m1, std1).pdf(d[i][step*j:step*(j+1)])).sum()
        if (G0_like>G1_like): # checich gausian is more likely
            est_gen_from[i][j]=0
            if(gen_from[i][j]==1):
                err+=1 # adderror if it miss
        else:
            est_gen_from[i][j]=1
            if(gen_from[i][j]==0):
                err+=1
    err/=(series_size/step)
    plt.figure(1)
    plt.title("Classifiction Error from L(Gk/dm)")
    plt.ylabel("Error")
    plt.xlabel("Series Lenght")
    plt.grid()
    plt.scatter(2**i,err)
#%% Maximum likelihood estam
# 3.
def generate2GaussiansSeries(n, k):
  """Generate 2 gaussians in an array randomized by bernulish"""
  ber = stats.bernoulli.rvs(0.5,size=n).astype(bool)
  D=[]
  for q in ber:
    if q:
      D.append(np.random.normal(m0, std0, 2**k))
    else:
      D.append(np.random.normal(m1, std1, 2**k))

  return D, ber
def kl_mvn(m0, s0, m1, s1):
    """
    computes KL divergence from a single Gaussian mu0,std0 to
    Gaussian mu1,std1. """
    return np.log(s1 / s0) + (s0 ** 2 + (m0 - m1) ** 2) / (2 * s1 ** 2) - 1 / 2


errors = dict()
series_size = 512
my_from_est_all = []
mu_stdG0 = []
mu_stdG1 = []

# expirements k 0..7
for k in range(0, 7):
    error = 0
    step = 2**k
    # generated = each points is maybe from g0 and maybe from g1 by bernula
    D, ber = generate2GaussiansSeries(series_size // step, k)
    D = np.array(D)
    my_from_est = []
    i0 = 0
    D_flat = D.flatten()
    for index, i in enumerate(range(0, series_size//step)):
        # check gausian is more likely

        g0_est = np.log(stats.norm(m0, std0).pdf(D_flat[step*i:step*(i+1)])).sum()
        g1_est = np.log(stats.norm(m1, std1).pdf(D_flat[step*i:step*(i+1)])).sum()
        my_from_est.append(g0_est > g1_est)
        error += int(my_from_est[-1] != ber[index])
        i0 = i
    my_from_est = np.array(my_from_est)
    mu_stdG0.append(stats.norm.fit(D[my_from_est].flatten()))
    mu_stdG1.append(stats.norm.fit(D[~my_from_est].flatten()))
    errors[str(k)] = error

kldg1=[]
kldg0=[]
# KL Divergance
for k in range(0,7):
  kldg0.append(kl_mvn(m0, std0, mu_stdG0[k][0], mu_stdG0[k][1]))
  kldg1.append(kl_mvn(m1, std1, mu_stdG1[k][0], mu_stdG1[k][1]))
print(errors)
plt.figure()
plt.title("Estimetor G0\G1 Error")
plt.ylabel("Error")
plt.xlabel("Series Lenght (k)")
plt.grid()
number_series = (series_size/np.array([2**k for k in range(0,7)]))
plt.scatter(np.arange(0,7) ,np.array(list(errors.values()))/number_series)
print(mu_stdG0)
print(mu_stdG1)
plt.figure(2)
plt.title("KL Divergence of G0")
plt.ylabel("Divergence")
plt.xlabel("Expirment")
plt.plot(kldg0)

plt.figure(3)
plt.title("KL Divergence of G1")
plt.ylabel("Divergence")
plt.xlabel("Expirment")
plt.plot(kldg1)
#%% Maximum likelihood estamation

mu_stdG0=np.zeros((7,2))
mu_stdG1=np.zeros((7,2))


g1_lenght=[]
g0_lenght=[]
lol=[]
for i in range (0,7):
 # for each experiment calulate the mean and std for each Generated G0 and G1 from likelod
    step=2**i
    G0k=np.array([])
    G1k=np.array([])
    for j in range(0,int(series_size/step)):
        #builds G1 and G0 for each experiment(i) based on the likelihood
        if (est_gen_from[i][j]==0):
            G0k=np.append(G0k,d[i][step*j:step*(j+1)])
        else:
             G1k=np.append(G1k,d[i][step*j:step*(j+1)])
        lol=np.append(lol,j)
        g1_lenght=np.append(g1_lenght,G1k.shape[0])
        g0_lenght=np.append(g0_lenght,G0k.shape[0])
        mu_stdG0[i]=stats.norm.fit(G0k)
        mu_stdG1[i]=stats.norm.fit(G1k)

    #%% KL Divergance
    kldg1=[]
    kldg0=[]
    for i in range (0,7):
        kldg0=np.append(kldg0,kl_mvn(m0, std0, mu_stdG0[i][0], mu_stdG0[i][1]))
        kldg1=np.append(kldg1,kl_mvn(m1, std1, mu_stdG1[i][0], mu_stdG1[i][1]))

plt.figure(2)
plt.title("KL Divergence of G0")
plt.ylabel("Divergence")
plt.xlabel("Expirment")
plt.plot(kldg0)

plt.figure(3)
plt.title("KL Divergence of G1")
plt.ylabel("Divergence")
plt.xlabel("Expirment")
plt.plot(kldg1)