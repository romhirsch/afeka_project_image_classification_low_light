# -- coding: utf-8 --
"""
Spyder Editor

This is a temporary script file.
"""
# from _future_ import print_function
import sys
import os

sys.path.append("C:\\Users\\spring1\\Desktop\\New folder (3)")

import numpy as np
import librosa
import os
import sys
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.utils import get_custom_objects
import matplotlib.pyplot as plt
import pickle
from scipy.stats import laplace
import plotly.graph_objects as go
import datetime
import re
from scipy.cluster.vq import vq, kmeans, whiten
np.set_printoptions(precision=10, suppress=True) # for compact output
from sklearn import cluster
# from python_speech_features import mfcc
# from python_speech_features import logfbank
# from python_speech_features import delta
import scipy.io.wavfile as wav
import pickle
import scipy.io
import re
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
def gen_multivariate_normal_clus(seed,dim,clus_std,clus_cov_mu,size):
    np.random.seed(seed=seed)
    clus_mu=np.random.uniform(0,clus_std,(dim))
    clus_cov_std=np.random.uniform(0,clus_cov_mu,(dim,dim))
    clus_cov_std_symm = (clus_cov_std + clus_cov_std.T)/2
    x=np.random.multivariate_normal(clus_mu,clus_cov_std_symm,size)
    return x
#%%
lab_t=[]
seed=42
dim=3
num_clus=2
clus_std_v=[1,6]
clus_cov_mu_v=[0.2,0.2]
size_v=np.array([100,50])
data=[]
lab=[]
for i,s in enumerate(size_v):
    lab+=[i]*s
for i in range(num_clus):
    if i==0:
        data=gen_multivariate_normal_clus(i,dim,clus_std_v[i],clus_cov_mu_v[i],size_v[i])
    else:
        all_data=np.concatenate((data,gen_multivariate_normal_clus(i,dim,clus_std_v[i],clus_cov_mu_v[i],size_v[i])),0)
df=pd.DataFrame(all_data)
df.loc[:,'cluster']=lab
df_gever=df
import matplotlib.pyplot as plt
import seaborn as sns


sns.scatterplot(x=df_gever[0], y=df_gever[1], hue=df_gever['cluster'])
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = df_gever[0]
y = df_gever[1]
z = df_gever[2]
ax.scatter(x=df_gever[0], y=df_gever[1], z=df_gever[2])

plt.show()


#%%
import re, seaborn as sns, numpy as np, pandas as pd, random
from pylab import *
from matplotlib.pyplot import plot, show, draw, figure, cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid", {'axes.grid' : False})

fig = plt.figure(figsize=(6,6))

ax = Axes3D(fig) # Method 1
# ax = fig.add_subplot(111, projection='3d') # Method 2

x = np.random.uniform(1,20,size=20)
y = np.random.uniform(1,100,size=20)
z = np.random.uniform(1,100,size=20)
df_gever.loc[df_gever['cluster']==0,'cluster'] = 'red'
df_gever.loc[df_gever['cluster']==1,'cluster'] = 'blue'


ax.scatter(df_gever[0], df_gever[1], df_gever[2], c=df_gever['cluster'], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()