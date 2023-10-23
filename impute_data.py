#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
import spektral
import matplotlib.pyplot as plt
import math
import scipy.sparse as sp
import random
import time

import warnings
warnings.filterwarnings('ignore')

import model
import func
import MLGAN_L5


# In[ ]:


env = func.env


# In[ ]:


st_dataset_normal,datamean,datastd = func.convertdataset_workday_weekend(env.path)
wrkd_dataset,wekd_dataset=func.split_workdayandweekend(st_dataset_normal)
adj = func.giveadj(env.adj_csvpath)
env.setadj(adj)
maskmatrixes = func.Maskmatrixes(env.rmdir)


# In[ ]:


mmdata = []
for i in env.missingrate:    
    mmdata.append(maskmatrixes[env.RandNUMS*i:env.RandNUMS*(i+1)])#mmdata为19个缺失区间的掩码矩阵组
    #mmdata.append(maskmatrixes[2*RandNUMS*i:2*RandNUMS*(i+1)])
#将

combvals = []#
tolcomvvals = []#
origin_dataset = wrkd_dataset[slice(int(0.95*len(wrkd_dataset)),-1)]#
#origin_dataset = wrkd_dataset[slice(int(0.85*len(wrkd_dataset)),int(0.95*len(wrkd_dataset)))]#目标数据集类型，分为周末或工作日
for i in mmdata:   
    RandNUMS_dataset = origin_dataset
    combval = []
    toldata = []
    tolmm = []
    for j in origin_dataset:
        for p in i:  
            k = np.expand_dims(j,axis=0)
            p = np.expand_dims(p,axis=0)
            combval.append((k,p))
            toldata.append(k)
            tolmm.append(p)
    tolcomdata = np.concatenate(toldata,axis=0)
    tolcommm = np.concatenate(tolmm,axis=0)
    tolcomvvals.append((tolcomdata,tolcommm))
    combvals.append(combval)
#combvals （origindata，mask）


# In[ ]:


inptest=combvals[18][0]
inptest


# In[ ]:


#load weight 
M5genl1 = model.Layer1MLGAN(env.timescale,env.spatialscale)
M5genl2 = model.Layer2MLGAN(env.timescale,env.spatialscale)
M5genl3 = model.Layer3MLGAN(env.timescale,env.spatialscale)
M5genl4 = model.Layer4MLGAN(env.timescale,env.spatialscale)
M5genl5 = model.Layer5MLGAN(env.timescale,env.spatialscale)
M5L1model  = model.MLGANcorrection(M5genl1)
M5L1model(inptest,env.ADJMatrix)#
M5L2model  = model.MLGANcorrection(M5genl2)
M5L2model(inptest,env.ADJMatrix)
M5L3model  = model.MLGANcorrection(M5genl3)
M5L3model(inptest,env.ADJMatrix)
M5L4model  = model.MLGANcorrection(M5genl4)
M5L4model(inptest,env.ADJMatrix)
M5L5model  = model.MLGANcorrection(M5genl5)
M5L5model(inptest,env.ADJMatrix)

M5L1model.load_weights(os.path.join(env.codepath,'MLGAN_Data\modelweight\L5GAN_NEW\corlayer1forM5\cor_wight'))
M5L2model.load_weights(os.path.join(env.codepath,'MLGAN_Data\modelweight\L5GAN_NEW\corlayer2forM5\cor_wight'))
M5L3model.load_weights(os.path.join(env.codepath,'MLGAN_Data\modelweight\L5GAN_NEW\corlayer3forM5\cor_wight'))
M5L4model.load_weights(os.path.join(env.codepath,'MLGAN_Data\modelweight\L5GAN_NEW\corlayer4forM5\cor_wight'))
M5L5model.load_weights(os.path.join(env.codepath,'MLGAN_Data\modelweight\L5GAN_NEW\corlayer5forM5\cor_wight'))



# In[ ]:


calmodel = MLGAN_L5.MLGAN(M5L1model,M5L2model,M5L3model,M5L4model,M5L5model)
_=calmodel(inptest,env.ADJMatrix)


# In[ ]:


origdata,_=inptest
origdata*datastd+datamean


# In[ ]:


calmodel(inptest,env.ADJMatrix)*datastd+datamean


# In[ ]:


def calcompareddata(ratioindex):
    
    calvals = tolcomvvals[ratioindex]
    caldata,calmm=calvals 

    oridata = (caldata*datastd + datamean).astype(np.int)

    resdata = None
    with tf.device('/CPU:0'):
        resdata = calmodel(calvals,env.ADJMatrix).numpy()
    
    cordata = resdata*(1-calmm) + caldata *calmm

    impdata = (cordata*datastd + datamean).astype(np.int)
    return oridata,impdata,calmm


# In[ ]:


df=pd.DataFrame(columns=['MAE','RMSE'])
for i in range(19):
    ori,imp,mm = calcompareddata(i)    
    newloc = func.returnloss(ori,imp,mm)
    print(newloc)
    df.loc[i+1]= newloc
df


# In[ ]:


#draw pic for data imputation
drawindex =18#missing rate
mmindex=8

inpdraw = combvals[drawindex][mmindex]
origdata_draw,mm_draw=inpdraw 


#
Z_draw =np.sum(origdata_draw,axis=0)
mm_draw = np.sum(mm_draw,axis=0)
index_sort=np.argsort(Z_draw.sum(axis=0))

#
Z_mask = Z_draw * mm_draw

#
Z_imp = calmodel(inpdraw,env.ADJMatrix)*(1-mm_draw) + Z_draw * mm_draw
Z_imp  = np.sum(Z_imp ,axis=0)
Z_gen = calmodel.activegan(inpdraw,env.ADJMatrix)
Z_gen =  np.sum(Z_gen ,axis=0)

#
Z_sort =Z_draw[:,index_sort]*datastd+datamean
Z_mask_sort =Z_mask[:,index_sort]*datastd+mm_draw[:,index_sort]*datamean
Z_imp_sort =Z_imp[:,index_sort]*datastd+datamean
Z_gen_sort = Z_gen[:,index_sort]*datastd+datamean

#
x = np.arange(0,Z_draw.shape[0],1)
y = np.arange(0,Z_draw.shape[1],1)


# In[ ]:


#Data imputation in the time dimension

timedim=45
tdim2x = np.arange(Z_sort.shape[0])
tdim2xmm = mm_draw[:,index_sort][:,timedim]
#tdim2xmm = np.array(tdim2xmm,dtype='int')
observed_time =np.where(tdim2xmm!=0)
imp_time = np.where(tdim2xmm == 0)
plt.figure(figsize=(14,6))


#plt.plot(dim2x,Z_mask_sort[:,timedim])
plt.plot(tdim2x,Z_sort[:,timedim],linewidth =1.2,label='origin data')

plt.scatter(tdim2x[imp_time],Z_sort[:,timedim][imp_time],marker = 'v',s=45,color = 'red',label='missing data',linewidths=2)
#plt.plot(tdim2x[imp_time],Z_sort[:,timedim][imp_time],label='missing data',linestyle='--',color = 'green',linewidth=2)

#plt.plot(tdim2x[imp_time],Z_imp_sort[:,timedim][imp_time],linewidth =2,color = 'blue',label='synthetic data')
plt.scatter(tdim2x[imp_time],Z_imp_sort[:,timedim][imp_time],marker = 'x',s=65,color = 'blue',label='synthetic data',linewidths=1.5)

#plt.scatter(tdim2x[observed_time],Z_sort[:,timedim][observed_time],marker = 'v',s=64,label='observed data')
plt.legend(fontsize = 20)
plt.yticks(fontsize=24,fontproperties = 'Times New Roman')
plt.xticks(fontsize=24,fontproperties = 'Times New Roman')
plt.show()


# In[ ]:


#Data imputation in the space dimension

spadim=11
tdim2y = np.arange(Z_sort.shape[1])
tdim2ymm = mm_draw[:,index_sort][spadim,:]
#tdim2xmm = np.array(tdim2xmm,dtype='int')
observed_sensor =np.where(tdim2ymm!=0)
imp_sensor = np.where(tdim2ymm == 0)
plt.figure(figsize=(24,8))

plt.scatter(tdim2y[imp_sensor],Z_imp_sort[spadim,:][imp_sensor],marker = 'x',s=64,label='synthetic data',linewidths=3)
plt.plot(tdim2y[imp_sensor],Z_imp_sort[spadim,:][imp_sensor],linestyle='--',linewidth =3,label='synthetic data')
#plt.scatter(tdim2y[imp_sensor],Z_sort[spadim,:][imp_sensor],marker = '+',s=128,label='missing data',linewidths=2)
#plt.plot(dim2x,Z_mask_sort[:,timedim])
#plt.plot(tdim2y,Z_sort[spadim,:])
plt.plot(tdim2y,Z_sort[spadim,:],linewidth =2,label='missing data')
#plt.scatter(tdim2y[observed_sensor],Z_sort[spadim,:][observed_sensor],marker = 'o',s=36,label='observed data')
plt.legend(fontsize = 24)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.show()


# In[ ]:


#origin data
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(8,8))
#ax = fig.gca(projection='3d')
ax = fig.add_axes(Axes3D(fig))
X, Y = np.meshgrid(y, x)
ax.plot_surface(X, Y, Z_sort,cmap='viridis')
ax.tick_params(axis='both',labelsize=16)
plt.show()


# In[ ]:


#destroyed data based on mask
fig=plt.figure(figsize=(8,8))
ax = fig.add_axes(Axes3D(fig))
X, Y = np.meshgrid(y, x)
ax.plot_surface(X, Y, Z_mask_sort,cmap='viridis')
ax.tick_params(axis='both',labelsize=16)
plt.show()


# In[ ]:


#data from gan module
fig=plt.figure(figsize=(8,8))
ax = fig.add_axes(Axes3D(fig))
X, Y = np.meshgrid(y, x)
ax.plot_surface(X, Y, Z_gen_sort,cmap='viridis')
plt.show()


# In[ ]:


#restored data from MLGAN
fig=plt.figure(figsize=(8,8))
ax = fig.add_axes(Axes3D(fig))
X, Y = np.meshgrid(y, x)
ax.plot_surface(X, Y, Z_imp_sort,cmap='viridis')
ax.tick_params(axis='both',labelsize=16)
plt.show()


# In[ ]:




