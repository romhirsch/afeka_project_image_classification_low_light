import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

"""
Yarom swissa 203675814
Rom hirsch 313288763
"""
#yarom id
m0 = 0.2
v0 = np.sqrt(2)
#rom id
m1 = 0.3
v1 = np.sqrt(1.5)
std0 = np.sqrt(v0)
std1 = np.sqrt(v1)
print(m0, std0)
print(m1, std1)

def generate2Gaussians(n):
  """Generate 2 gaussians in an array randomized by bernulish"""
  ber = stats.bernoulli.rvs(0.5,size=n).astype(bool)
  g0 = np.random.normal(m0, std0, n)
  g1 = np.random.normal(m1, std1, n)

  generated = np.zeros(n)
  generated[ber] = g0[ber]
  generated[~ber] = g1[~ber]

  return generated, ber

  # plt.figure(figsize=(10,5))
  # plt.title(f'G0 m={m0}, v={v0}')
  # sigma = np.sqrt(v0)
  # x = np.linspace(m0 - 3*sigma, m0 + 3*sigma, 100)
  # plt.plot(x, stats.norm.pdf(x, m0, sigma))
  # plt.xlabel('Values')

  # plt.figure(figsize=(10,5))
  # plt.title(f'G1 m={m1}, v={v1}')
  # sigma = np.sqrt(v1)
  # x = np.linspace(m1 - 3*sigma, m1 + 3*sigma, 100)
  # plt.plot(x, stats.norm.pdf(x, m1, sigma))
  # plt.xlabel('Values')

  # plt.figure(figsize=(10,5))
  # plt.title(f'Generated')
  # x = plt.hist(generated, bins=100)
  # x = plt.hist(generated, bins=100)
  # plt.xlabel('Values')
#1.
generated, ber = generate2Gaussians(1024)
moe, v0e = stats.norm.fit(generated[ber])
m1e, v1e = stats.norm.fit(generated[~ber])
m0String = 'mean0_estimate'
v0String = 'std0_estimate'
m1String = 'mean1_estimate'
v1String = 'std1_estimate'
dict_res = {m0String:[], v0String:[],
            m1String:[], v1String:[],
            "n":[]}

#2.
n = [64, 256, 4096, 16384]
for i in n:
  generated, ber = generate2Gaussians(i)
  m0e, v0e = stats.norm.fit(generated[ber])
  m1e, v1e = stats.norm.fit(generated[~ber])
  dict_res[m0String].append(m0e)
  dict_res[v0String].append(v0e)
  dict_res[m1String].append(m1e)
  dict_res[v1String].append(v1e)
  dict_res['n'].append(i)

df = pd.DataFrame(dict_res)
df[m0String+'_err']=np.abs(df[m0String]-m0)
df[v0String+'_err']=np.abs(df[v0String]-std0)
df[m1String+'_err']=np.abs(df[m1String]-m1)
df[v1String+'_err']=np.abs(df[v1String]-std1)

print(df)


plt.figure(); plt.title('error for m0 estimation'); plt.xlabel('n'); plt.ylabel('abs(m0-m0_estimated)') ;plt.plot(dict_res['n'], df[m0String+'_err'])
plt.figure(); plt.title('error for v0 estimation'); plt.xlabel('n'); plt.ylabel('abs(v0-v0_estimated)') ;plt.plot(dict_res['n'], df[m1String+'_err'])
plt.figure(); plt.title('error for m1 estimation'); plt.xlabel('n'); plt.ylabel('abs(m1-m1_estimated)') ;plt.plot(dict_res['n'], df[v0String+'_err'])
plt.figure(); plt.title('error for v1 estimation'); plt.xlabel('n'); plt.ylabel('abs(v1-v1_estimated)') ;plt.plot(dict_res['n'], df[v1String+'_err'])

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
#3.
def kl_mvn(m0, s0, m1, s1):
 """
 computes KL divergence from a single Gaussian mu0,std0 to
 Gaussian mu1,std1. """
 return np.log(s1/s0)+(s0**2+(m0-m1)**2)/(2*s1**2)-1/2

errors = dict()
series_size = 512
my_from_est_all = []
mu_stdG0 = []
mu_stdG1 = []
dict_res = {m0String:[], v0String:[],
            m1String:[], v1String:[],
            "k":[]}
# expirements k 0..7
for k in range(0, 7):
    error = 0
    step = 2**k
    # generated = each points is maybe from g0 and maybe from g1 by bernula
    D, ber = generate2GaussiansSeries(512, k)
    D = np.array(D)
    my_from_est = []
    i0 = 0
    D_flat = D.flatten()
    for index, i in enumerate(range(0, series_size)):
        # check gausian is more likely
        g0_est = np.log(stats.norm(m0, std0).pdf(D_flat[step*i:step*(i+1)])).sum()
        g1_est = np.log(stats.norm(m1, std1).pdf(D_flat[step*i:step*(i+1)])).sum()
        my_from_est.append(g0_est > g1_est)
        error += int(my_from_est[-1] != ber[index])
        i0 = i
    my_from_est = np.array(my_from_est)
    mu_stdG0 = stats.norm.fit(D[my_from_est].flatten())
    mu_stdG1 = stats.norm.fit(D[~my_from_est].flatten())
    dict_res[m0String].append(mu_stdG0[0])
    dict_res[v0String].append(mu_stdG0[1])
    dict_res[m1String].append(mu_stdG1[0])
    dict_res[v1String].append(mu_stdG1[1])
    dict_res['k'].append(k)

    errors[str(k)] = error
df_est = pd.DataFrame(dict_res)
print(df_est)

kldg1=[]
kldg0=[]
# KL Divergance
for k in range(0,7):
  kldg0.append(kl_mvn(m0, std0, df_est.loc[k, m0String], df_est.loc[k, v0String]))
  kldg1.append(kl_mvn(m1, std1, df_est.loc[k, m1String], df_est.loc[k, v1String]))
print(errors)
plt.figure()
plt.title("Estimetor G0\G1 Error")
plt.ylabel("Error")
plt.xlabel("Series Lenght (k)")
plt.grid()
number_series = (series_size/np.array([2**k for k in range(0,7)]))
plt.scatter([2**k for k in range(0,7)] ,np.array(list(errors.values()))/number_series, color=['r', 'g', 'b', 'k', 'm', 'y','c'])
plt.legend()
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
