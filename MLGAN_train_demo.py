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


# combine the flow data with mask matrixes
mmdata = []
for i in env.missingrate:    
    mmdata.append(maskmatrixes[env.RandNUMS*i:env.RandNUMS*(i+1)])#there are 19 types of mask matrixes based on missing rate
    #mmdata.append(maskmatrixes[2*RandNUMS*i:2*RandNUMS*(i+1)])
#   
mrfordata_data = []
dataformr_data = []
#we test workday data in this demo
origin_dataset = wrkd_dataset[slice(0,int(0.95*len(wrkd_dataset)))]#dataset will be divided into  into train and test data
#test_dataset = wrkd_dataset[int(0.95*len(wrkd_dataset)):]
for i in mmdata:   
    RandNUMS_dataset = origin_dataset
    mrfordata = []
    dataformr = []
    for j in origin_dataset:
        for p in i:            
            mrfordata.append(p)
            dataformr.append(j)
    mrfordata_data.append(mrfordata)
    dataformr_data.append(dataformr)

#convert data into dataset(keras)
mmr_datasets = []
with tf.device('/device:GPU:0'):
    for i in env.missingrate :
        mrfordata_dataset =  tf.data.Dataset.from_tensor_slices(np.array(mrfordata_data[i]))
        dataformr_dataset =  tf.data.Dataset.from_tensor_slices(np.array(dataformr_data[i]))
        mmr_dataset = tf.data.Dataset.zip((dataformr_dataset,mrfordata_dataset)).shuffle(len(mrfordata_data[1])).batch(env.batch_size)
        mmr_datasets.append(mmr_dataset)


# In[ ]:


#Classify datasets based on missingrate
#5layer model 0-18：0-3，4-7，8-11,12-15,16-18
layer1dataforM5 = mmr_datasets[0]
for i in mmr_datasets[1:4]:
    layer1dataforM5 = layer1dataforM5.concatenate(i)

layer2dataforM5 = mmr_datasets[4]
for i in mmr_datasets[5:8]:
    layer2dataforM5 = layer2dataforM5.concatenate(i)

layer3dataforM5 = mmr_datasets[8]
for i in mmr_datasets[9:12]:
    layer3dataforM5 = layer3dataforM5.concatenate(i)
    
layer4dataforM5 = mmr_datasets[12]
for i in mmr_datasets[13:16]:
    layer4dataforM5 = layer4dataforM5.concatenate(i)
    
layer5dataforM5 = mmr_datasets[16]
for i in mmr_datasets[17:]:
    layer5dataforM5 = layer5dataforM5.concatenate(i)    
    
#shuffle to disrupting data
layer1dataforM5 = layer1dataforM5.shuffle(buffer_size = len(list(mmr_datasets[0]))*(5+1))
layer1dataforM5 = layer1dataforM5.concatenate(layer1dataforM5)
layer2dataforM5 = layer2dataforM5.shuffle(buffer_size = len(list(mmr_datasets[0]))*(5+1))
layer2dataforM5 = layer2dataforM5.concatenate(layer2dataforM5)
layer3dataforM5 = layer3dataforM5.shuffle(buffer_size = len(list(mmr_datasets[0]))*(5+1))
layer3dataforM5 = layer3dataforM5.concatenate(layer3dataforM5)
layer4dataforM5 = layer4dataforM5.shuffle(buffer_size = len(list(mmr_datasets[0]))*(5+1))
layer4dataforM5 = layer4dataforM5.concatenate(layer4dataforM5)
layer5dataforM5 = layer5dataforM5.shuffle(buffer_size = len(list(mmr_datasets[0]))*(5+1))
layer5dataforM5 = layer5dataforM5.concatenate(layer5dataforM5)


# In[ ]:


#we train the MAGAN layer by layer
save_path = os.path.join(env.codepath,'MLGAN_Data','modelweight')


# In[ ]:


#we train the gan module first
genlayer1forM5 = model.Layer1MLGAN(288,277)#layer1 model
discforM5 =model.MLGANDiscriminator()


# In[ ]:


func.train(genlayer1forM5,discforM5,layer1dataforM5,env.gan_epochs)

#Save model parameters in a specific location
#genlayer1_path = os.path.join(save_path,'L5GAN','genlayer1forM5','gen_wight')
#disclayer1_path = os.path.join(save_path,'L5GAN','disclayer1forM5','disc_wight')
#genlayer1forM5.save_weights(genlayer1_path)
#discforM5.save_weights(disclayer1_path)


# In[ ]:


genlayer2forM5 = model.Layer2MLGAN(288,277,fiternums=12,paddingshape=24)#layer2 model

#Layer 1 shares parameter information of the input and sampling layers to Layer 2 
genlayer2forM5.inherit(genlayer1forM5)
genlayer2forM5.Downsampler1.trainable = False
genlayer2forM5.Upsampler1.trainable = False
#discforM5 =model.MLGANDiscriminator()


# In[ ]:


func.train(genlayer2forM5,discforM5,layer2dataforM5,env.gan_epochs)
#genlayer2_path = os.path.join(save_path,'L5GAN','genlayer2forM5','gen_wight')
#disclayer2_path = os.path.join(save_path,'L5GAN','disclayer2forM5','disc_wight')
#genlayer2forM5.save_weights(genlayer2_path)
#discforM5.save_weights(disclayer2_path)


# In[ ]:


genlayer3forM5 = model.Layer3MLGAN(288,277,fiternums=12,paddingshape=48)#layer3 model

genlayer3forM5.inherit(genlayer2forM5)
genlayer3forM5.Downsampler1.trainable = False
genlayer3forM5.Upsampler1.trainable = False
genlayer3forM5.Downsampler2.trainable = False
genlayer3forM5.Upsampler2.trainable = False


# In[ ]:


func.train(genlayer3forM5,discforM5,layer3dataforM5,env.gan_epochs)
#genlayer3_path = os.path.join(save_path,'L5GAN','genlayer3forM5','gen_wight')
#disclayer3_path = os.path.join(save_path,'L5GAN','disclayer3forM5','disc_wight')
#genlayer3forM5.save_weights(genlayer3_path)
#discforM5.save_weights(disclayer3_path)


# In[ ]:


genlayer4forM5 = model.Layer4MLGAN(288,277,fiternums=12,paddingshape=72)#layer4 model

genlayer4forM5.inherit(genlayer3forM5)
genlayer4forM5.Downsampler1.trainable = False
genlayer4forM5.Upsampler1.trainable = False
genlayer4forM5.Downsampler2.trainable = False
genlayer4forM5.Upsampler2.trainable = False
genlayer4forM5.Downsampler3.trainable = False
genlayer4forM5.Upsampler3.trainable = False


# In[ ]:


func.train(genlayer4forM5,discforM5,layer4dataforM5,env.gan_epochs)
#genlayer4_path = os.path.join(save_path,'L5GAN','genlayer4forM5','gen_wight')
#disclayer4_path = os.path.join(save_path,'L5GAN','disclayer4forM5','disc_wight')
#genlayer4forM5.save_weights(genlayer4_path)
#discforM5.save_weights(disclayer4_path)


# In[ ]:


genlayer5forM5 = model.Layer5MLGAN(288,277,fiternums=12,paddingshape=96)#五层模型

genlayer5forM5.inherit(genlayer4forM5)
genlayer5forM5.Downsampler1.trainable = False
genlayer5forM5.Upsampler1.trainable = False
genlayer5forM5.Downsampler2.trainable = False
genlayer5forM5.Upsampler2.trainable = False
genlayer5forM5.Downsampler3.trainable = False
genlayer5forM5.Upsampler3.trainable = False
genlayer5forM5.Downsampler4.trainable = False
genlayer5forM5.Upsampler4.trainable = False


# In[ ]:


func.train(genlayer5forM5,discforM5,layer5dataforM5,env.gan_epochs)
#genlayer5_path = os.path.join(save_path,'L5GAN','genlayer5forM5','gen_wight')
#disclayer5_path = os.path.join(save_path,'L5GAN','disclayer5forM5','disc_wight')
#genlayer5forM5.save_weights(genlayer5_path)
#discforM5.save_weights(disclayer5_path)


# In[ ]:


#we train the cor module then


# In[ ]:


#train correction layer
MLGANcorL1M5 = model.MLGANcorrection(genlayer1forM5)
MLGANcorL1M5.gan.trainning = False


# In[ ]:


func.trainMLGAN(MLGANcorL1M5,layer1dataforM5,env.cor_epochs)
#corlayer1_path = os.path.join(save_path,'L5GAN','corlayer1forM5','cor_wight')
#MLGANcorL1M5.save_weights(corlayer1_path)


# In[ ]:


MLGANcorL2M5 = model.MLGANcorrection(genlayer2forM5)
MLGANcorL2M5.gan.trainning = False


# In[ ]:


func.trainMLGAN(MLGANcorL2M5,layer2dataforM5,env.cor_epochs)
#corlayer1_path = os.path.join(save_path,'L5GAN','corlayer1forM5','cor_wight')
#MLGANcorL1M5.save_weights(corlayer1_path)


# In[ ]:


MLGANcorL3M5 = model.MLGANcorrection(genlayer3forM5)
MLGANcorL3M5.gan.trainning = False


# In[ ]:


func.trainMLGAN(MLGANcorL3M5,layer3dataforM5,env.cor_epochs)
#corlayer3_path = os.path.join(save_path,'L5GAN','corlayer3forM5','cor_wight')
#MLGANcorL3M5.save_weights(corlayer3_path)


# In[ ]:


MLGANcorL4M5 = model.MLGANcorrection(genlayer4forM5)
MLGANcorL4M5.gan.trainning = False


# In[ ]:


func.trainMLGAN(MLGANcorL4M5,layer4dataforM5,env.cor_epoc)
#corlayer4_path = os.path.join(save_path,'L5GAN','corlayer4forM5','cor_wight')
#MLGANcorL4M5.save_weights(corlayer4_path)


# In[ ]:


MLGANcorL5M5 = model.MLGANcorrection(genlayer5forM5)
MLGANcorL5M5.gan.trainning = False


# In[ ]:


func.trainMLGAN(MLGANcorL5M5,layer5dataforM5,env.cor_epoc)
#corlaye51_path = os.path.join(save_path,'L5GAN','corlayer5forM5','cor_wight')
#MLGANcorL5M5.save_weights(corlayer5_path)


# In[ ]:


MLGANmodel_demo = MLGAN_L5.MLGAN(MLGANcorL1M5,MLGANcorL2M5,MLGANcorL3M5,MLGANcorL4M5,MLGANcorL5M5)#MLGAN layer5


# In[ ]:


#test data
inptest = next(iter(layer5dataforM5))
MLGANmodel_demo(inptest,env.ADJMatrix)*datastd+datamean


# In[ ]:




