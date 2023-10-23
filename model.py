import os
import pandas as pd
import numpy as np
import tensorflow as tf
import spektral
import matplotlib.pyplot as plt
import math
import scipy.sparse as sp
import random

class TemporalFeatureMachine(tf.keras.layers.Layer):
    def __init__(self,spatialdim):
        super(TemporalFeatureMachine,self).__init__()
        #self.MyBiGRU= tf.keras.layers.Bidirectional(tf.keras.layers.GRU(spatialdim,return_sequences=True),merge_mode='ave')
        self.MyBiGRU= tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64,return_sequences=True),merge_mode='sum')
        #self.activelayer = tf.keras.layers.Activation('relu')
        self.TFDense = tf.keras.layers.Dense(spatialdim,activation = 'relu',name = 'TFDense')
        #self.TFDense = tf.keras.layers.Dense(spatialdim,name = 'TFDense')
    def call(self,inputs):
        x = inputs
        bilstm_1 = self.MyBiGRU(x)
        #act = self.activelayer(bilstm_1)
        res = self.TFDense( bilstm_1)
        return  res
    
class SpatialFeatureMachine(tf.keras.layers.Layer):
    def __init__(self,timedim):
        super(SpatialFeatureMachine,self).__init__()
        #self.MyGCN = spektral.layers.GCNConv(timedim,activation='relu',name='mygcn1')
        self.MyGCN = spektral.layers.GCNConv(64,activation='relu',name='mygcn1')
        self.SFDense = tf.keras.layers.Dense(timedim,activation = 'relu',name = 'SFDense')
        #self.SFDense = tf.keras.layers.Dense(timedim,name = 'SFDense')
    def call(self,inputs):
        x,a = inputs
        x = tf.transpose(x,perm=[0,2,1])#默认顺序为时间/空间，因此在使用GCN提取特征时需要交换位置
        gcn_1 = self.MyGCN([x,a])
        res = self.SFDense(gcn_1)
        return res



class Downsampler(tf.keras.layers.Layer):
    def __init__(self,fiters,rateindex,samplername):
        super(Downsampler,self).__init__()
        self.Conv2D_1 = tf.keras.layers.Conv2D(fiters*rateindex,3,strides=(1,1),#activation = 'relu',
                                               padding = 'same',name =samplername+'Conv2D_1')
        self.Conv2D_2 = tf.keras.layers.Conv2D(fiters*rateindex*2,3,strides=(1,1),#activation = 'relu',
                                               padding = 'same',name =samplername+'Conv2D_2')
        self.Maxpool = tf.keras.layers.MaxPool2D(pool_size = (2,2),strides=(2, 2))
    
    def call(self,inputs):#采样器仅处理规则数据（batchsize,f1dim,f2dim,dims）    
        conv2d_1 = self.Conv2D_1(inputs)
        conv2d_2 = self.Conv2D_2(conv2d_1)
        
        relu = tf.keras.layers.ReLU()(conv2d_2)
        maxpool = self.Maxpool(relu)
        #return conv2d_1,conv2d_2,maxpool
        return maxpool

class Upsampler(tf.keras.layers.Layer):#与下采样器不同，上采样器不仅需要与同级下采样器保持结构类似
    #，也要确保与下采样器的输入数据维度相同(输出数据需要与下采样器输入合并)
    def __init__(self,fiters,rateindex,paddingdatalastshape,samplername):#paddingdatalastshape为上次采样器输入数据的末尾轴维度按照实验设计进行调整（通常不受目标数据时空轴的影响，除非模型自身有要求）
        super(Upsampler,self).__init__()
        self.Upsample = tf.keras.layers.UpSampling2D(size = (2,2))
        #Conv2DTranspose的fiter为输出的维度因此结构与下采样器有所不同
        self.DeConv2D_1 = tf.keras.layers.Conv2DTranspose(fiters*rateindex, 3,strides=(1,1),#activation = 'relu',
                                               padding = 'same',name =samplername+'DeConv2D_1')
        self.DeConv2D_2 = tf.keras.layers.Conv2DTranspose(paddingdatalastshape,3,strides=(1,1),#activation = 'relu',
                                               padding = 'same',name =samplername+'DeConv2D_2')
      
   #采样器仅处理规则数据（batchsize,f1dim,f2dim,dims）
   
    def call(self,inputs,paddingdata):   #paddingdata_shape为同等级下采样器的输入数据，确保数据维度相当
        upsample = self.Upsample(inputs)
        upshape = upsample.shape
        padshape = paddingdata.shape
        pad_upsample = tf.pad(upsample,[[0,0],[0,padshape[1]-upshape [1]],
                                       [0,padshape[2]-upshape [2]],[0,0]],"CONSTANT")
       
        #tf.pad作用为扩展数据不同轴的维度，模型采用了末尾扩展方式
        #对应的数据结构在f1dim,f2dim两轴进行扩展,标准格式为[起点扩展，终点扩展]对应为[0，对照数据轴维度-被扩展数据轴维度]
        #需要注意的是扩展最好在upsample后紧接着进行，以确保扩展维的数据参与优化，防止上采样后的数据因扩展维产生偏移
        deconv2d_1 = self.DeConv2D_1(pad_upsample )
        
        deconv2d_2 = self.DeConv2D_2(deconv2d_1)
        #norm = tf.keras.layers.BatchNormalization()(deconv2d_2)
        #relu = tf.keras.layers.ReLU()(deconv2d_2)
        res =deconv2d_2+paddingdata
        #return  upsample,pad_upsample,deconv2d_1,deconv2d_2,res
        return  res

#时空特征结构
#基础单层结构
class Layer1MLGAN(tf.keras.Model):
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=2):
        super(Layer1MLGAN,self).__init__()
        #GAN过程一直保持更新
        self.Textractor = TemporalFeatureMachine(spatialdim)
        self.Sextractor = SpatialFeatureMachine(timedim)
        self.GANDenselayer = tf.keras.layers.Dense(1,name='GANDense')
        #self.GANDenselayer2 = tf.keras.layers.Dense(1,name='GANDense2')
        #GAN过程分层更新
        self.Downsampler1 = Downsampler(fiternums,1,'Downsampler1')#2-->12->24
        self.Upsampler1 = Upsampler(fiters=fiternums,rateindex=1,paddingdatalastshape=paddingshape,samplername='Upsampler1') #24-->12->2++>2            
    #特征合并
    
    def CombineFeature(self,tfeature,sfeature):
        tfeature = tf.expand_dims(tfeature,axis=-1)
        sfeature = tf.expand_dims(tf.transpose(sfeature,perm=[0,2,1]),axis=-1)
        comfeature= tf.concat([tfeature,sfeature],axis=-1)
        return comfeature
    #特征提取
        
    def ExtractorFeature(self,mminp,adj_m):        
        #特征提取
        temporalf = self.Textractor(mminp)
        spatialf = self.Sextractor([mminp,adj_m])
        comf = self.CombineFeature(temporalf,spatialf)#特征合并
        return comf
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        upf1 = self.Upsampler1(downf1,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        #gandense2= self.GANDenselayer2(gandense1)
        res = tf.reduce_sum(gandense,axis = -1)           
        return res



class Layer2MLGAN(Layer1MLGAN):#2层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=24):
        super(Layer2MLGAN,self).__init__(timedim,spatialdim,12,2)
        
        #GAN过程分层更新
        self.Downsampler2 = Downsampler(fiternums,2,'Downsampler2')#2 -->12->24 -->24->48
        self.Upsampler2 = Upsampler(fiters=fiternums,rateindex=2,paddingdatalastshape=24,samplername='Upsampler2')#48-->24->24++>24-->12->2 ++>2      
    def inherit(self, layer1gan):#继承权重参数
        self.Textractor = layer1gan.Textractor
        self.Sextractor = layer1gan.Sextractor
        self.GANDenselayer = layer1gan.GANDenselayer      
        self.Downsampler1 = layer1gan.Downsampler1
        self.Upsampler1 = layer1gan.Upsampler1 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        upf2 = self.Upsampler2(downf2,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res
    
class Layer3MLGAN(Layer2MLGAN):#3层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=48):
        super(Layer3MLGAN,self).__init__(timedim,spatialdim,12,24)        
        #GAN过程分层更新
        self.Downsampler3 = Downsampler(fiternums,3,'Downsampler3')#2 -->12->24-->24->48-->36->72
        self.Upsampler3 = Upsampler(fiters=fiternums,rateindex=3,paddingdatalastshape=48,samplername='Upsampler3')#72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer2gan):#继承权重参数
        self.Textractor = layer2gan.Textractor
        self.Sextractor = layer2gan.Sextractor
        self.GANDenselayer = layer2gan.GANDenselayer      
        self.Downsampler1 = layer2gan.Downsampler1
        self.Upsampler1 = layer2gan.Upsampler1 
        self.Downsampler2 = layer2gan.Downsampler2
        self.Upsampler2 = layer2gan.Upsampler2 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        upf3 = self.Upsampler3(downf3,downf2)
        upf2=self.Upsampler2(upf3 ,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res


class Layer4MLGAN(Layer3MLGAN):#4层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=72):
        super(Layer4MLGAN,self).__init__(timedim,spatialdim,12,48)        
        #GAN过程分层更新
        self.Downsampler4 = Downsampler(fiternums,4,'Downsampler4')#2 -->12->24-->24->48-->36->72-->48->96
        self.Upsampler4 = Upsampler(fiters=fiternums,rateindex=4,paddingdatalastshape=72,samplername='Upsampler4')#96-->48->72++>72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer3gan):#继承权重参数
        self.Textractor = layer3gan.Textractor
        self.Sextractor = layer3gan.Sextractor
        self.GANDenselayer = layer3gan.GANDenselayer      
        self.Downsampler1 = layer3gan.Downsampler1
        self.Upsampler1 = layer3gan.Upsampler1 
        self.Downsampler2 = layer3gan.Downsampler2
        self.Upsampler2 = layer3gan.Upsampler2 
        self.Downsampler3 = layer3gan.Downsampler3
        self.Upsampler3 = layer3gan.Upsampler3 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        downf4 = self.Downsampler4(downf3)
        upf4 = self.Upsampler4(downf4,downf3)
        upf3 = self.Upsampler3(upf4,downf2)
        upf2=self.Upsampler2(upf3,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res

class Layer5MLGAN(Layer4MLGAN):#5层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=96):
        super(Layer5MLGAN,self).__init__(timedim,spatialdim,12,72)        
        #GAN过程分层更新
        self.Downsampler5 = Downsampler(fiternums,5,'Downsampler4')#2 -->12->24-->24->48-->36->72-->48->96-->60->120
        self.Upsampler5 = Upsampler(fiters=fiternums,rateindex=5,paddingdatalastshape=96,samplername='Upsampler4')#120-->60->96++>96-->48->72++>72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer4gan):#继承权重参数
        self.Textractor = layer4gan.Textractor
        self.Sextractor = layer4gan.Sextractor
        self.GANDenselayer = layer4gan.GANDenselayer      
        self.Downsampler1 = layer4gan.Downsampler1
        self.Upsampler1 = layer4gan.Upsampler1 
        self.Downsampler2 = layer4gan.Downsampler2
        self.Upsampler2 = layer4gan.Upsampler2 
        self.Downsampler3 = layer4gan.Downsampler3
        self.Upsampler3 = layer4gan.Upsampler3 
        self.Downsampler4 = layer4gan.Downsampler4
        self.Upsampler4 = layer4gan.Upsampler4 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        downf4 = self.Downsampler4(downf3)
        downf5 = self.Downsampler5(downf4)
        upf5 = self.Upsampler5(downf5,downf4)
        upf4 = self.Upsampler4(upf5,downf3)
        upf3 = self.Upsampler3(upf4,downf2)
        upf2=self.Upsampler2(upf3,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res

class Layer6MLGAN(Layer5MLGAN):#6层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=120):
        super(Layer6MLGAN,self).__init__(timedim,spatialdim,12,96)        
        #GAN过程分层更新#
        self.Downsampler6 = Downsampler(fiternums,6,'Downsampler4')#2 -->12->24-->24->48-->36->72-->48->96-->60->120-->72->144
        self.Upsampler6 = Upsampler(fiters=fiternums,rateindex=6,paddingdatalastshape=120,samplername='Upsampler4')#144-->72->120++>120-->60->96++>96-->48->72++>72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer5gan):#继承权重参数
        self.Textractor = layer5gan.Textractor
        self.Sextractor = layer5gan.Sextractor
        self.GANDenselayer = layer5gan.GANDenselayer      
        self.Downsampler1 = layer5gan.Downsampler1
        self.Upsampler1 = layer5gan.Upsampler1 
        self.Downsampler2 = layer5gan.Downsampler2
        self.Upsampler2 = layer5gan.Upsampler2 
        self.Downsampler3 = layer5gan.Downsampler3
        self.Upsampler3 = layer5gan.Upsampler3 
        self.Downsampler4 = layer5gan.Downsampler4
        self.Upsampler4 = layer5gan.Upsampler4 
        self.Downsampler5 = layer5gan.Downsampler5
        self.Upsampler5 = layer5gan.Upsampler5 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        downf4 = self.Downsampler4(downf3)
        downf5 = self.Downsampler5(downf4)
        downf6 = self.Downsampler6(downf5)
        upf6 = self.Upsampler6(downf6,downf5)
        upf5 = self.Upsampler5(upf6 ,downf4)
        upf4 = self.Upsampler4(upf5,downf3)
        upf3 = self.Upsampler3(upf4,downf2)
        upf2=self.Upsampler2(upf3,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res

class MLGANDiscriminator(tf.keras.Model):
    def __init__(self):
        super(MLGANDiscriminator,self).__init__()
        self.Conv1 = tf.keras.layers.Conv2D(24,3,strides=(2,2),padding = 'same')      
        self.Conv2 = tf.keras.layers.Conv2D(64,3,strides=(2,2),padding = 'same')
        self.Conv3 = tf.keras.layers.Conv2D(128,3,strides=(2,2),padding = 'same',activation = 'relu')
        self.Dense5 = tf.keras.layers.Dense(128)
        self.Dense6 = tf.keras.layers.Dense(64)
        self.Dense7 = tf.keras.layers.Dense(1,activation = 'sigmoid')
        #self.Dense7 = tf.keras.layers.Dense(1)#,activation = 'sigmoid')

    def call(self,inp):
        inp = tf.expand_dims(inp,axis=-1)#目标数据都是（batch_size,f1dim.f2dim),需要扩展轴
        conv1 = self.Conv1(inp) 
        drop1 = tf.keras.layers.Dropout(0.2)(conv1 )       
        conv2= self.Conv2(drop1)   
        drop2 = tf.keras.layers.Dropout(0.2)(conv2 )               
        conv3 = self.Conv3(drop2)              
        flat4 = tf.keras.layers.Flatten()(conv3 )
        dense5 = self.Dense5(flat4)
        dense6 = self.Dense6(dense5)
        res = self.Dense7(dense6)
        return res
'''

class MLGANDiscriminator(tf.keras.Model):
    def __init__(self):
        super(MLGANDiscriminator,self).__init__()
        self.Conv1 = tf.keras.layers.Conv2D(24,2,strides=(1,1),padding = 'same',activation = 'relu')
        self.Maxp1 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides = (2,2))
        self.Conv2 = tf.keras.layers.Conv2D(48,2,strides=(1,1),padding = 'same',activation = 'relu')
        self.Maxp2 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides = (2,2))
        self.Conv3 = tf.keras.layers.Conv2D(64,2,strides=(1,1),padding = 'same',activation = 'relu')
        self.Maxp3 = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides = (2,2))
        self.Conv4 = tf.keras.layers.Conv2D(96,2,strides=(1,1),padding = 'same',activation = 'relu')
        self.Dense5 = tf.keras.layers.Dense(128,activation = 'relu')
        self.Dense6 = tf.keras.layers.Dense(64,activation = 'relu')
        #self.Dense7 = tf.keras.layers.Dense(1)
        self.Dense7 = tf.keras.layers.Dense(1,activation = 'softmax')
    def call(self,inp):
        inp = tf.expand_dims(inp,axis=-1)#目标数据都是（batch_size,f1dim.f2dim),需要扩展轴
        conv1 = self.Conv1(inp)
        maxp1 = self.Maxp1(conv1)
        conv2= self.Conv2(maxp1)
        norm2 = tf.keras.layers.BatchNormalization()(conv2)
        maxp2 = self.Maxp2(norm2)
        conv3 = self.Conv3(maxp2)
        norm3 = tf.keras.layers.BatchNormalization()(conv3)
        maxp3 = self.Maxp3(norm3)
        conv4 = self.Conv4(maxp3)
        norm4 = tf.keras.layers.BatchNormalization()(conv4)
        flat4 = tf.keras.layers.Flatten()(norm4)
        dense5 = self.Dense5(flat4)
        dense6 = self.Dense6(dense5)
        res = self.Dense7(dense6)
        return res
'''



class MLGANcorrection(tf.keras.Model):
    def __init__(self,mlgan):
        super(MLGANcorrection,self).__init__()
        self.gan = mlgan
        self.gan.trainable = False
        #仅使用重建损失更新
        #self.correctionlayer1= tf.keras.layers.Dense(64,name = 'correctionlayer')
       #self.correctionlayer2= tf.keras.layers.Dense(1,name = 'correctionlayer')
        self.correctionlayer = tf.keras.layers.Dense(277,name = 'correctionlayer')
    def call(self,inp,adj_m):
        syntheticdata = self.gan(inp,adj_m)
        #data_adddim = tf.expand_dims(syntheticdata,axis=-1)
        data_adddim =  syntheticdata
        correctiondata =self.correctionlayer(data_adddim)
        #correctiondata1 =self.correctionlayer1(data_adddim)
        #correctiondata2 =self.correctionlayer2(correctiondata1)
        #res = tf.reduce_sum(correctiondata2,axis=-1)
        res = correctiondata
        return res

'''
class MLGANcorrection(tf.keras.Model):
    def __init__(self,mlgan):
        super(MLGANcorrection,self).__init__()
        self.gan = mlgan
        self.gan.trainable = False
        #仅使用重建损失更新
        self.correctionlayer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32,1,1,padding='same',activation = 'relu',name = 'corconv1'),
            #tf.keras.layers.Conv2D(64,1,1,padding='same',activation = 'relu',name = 'corconv2'),
            #tf.keras.layers.Dense(128,name = 'cordense3',activation = 'relu'),
            tf.keras.layers.Dense(64,name = 'cordense2',activation = 'relu'),
            tf.keras.layers.Dense(32,name = 'cordense3'),
            tf.keras.layers.Dense(1,name ='cordense4')])
        
    def call(self,inputs,adj_m):
        syntheticdata = self.gan(inputs,adj_m)
        data_adddim = tf.expand_dims(syntheticdata,axis=-1)
        correctiondata =self.correctionlayer(data_adddim)
        res = tf.reduce_sum(correctiondata,axis=-1)
        return res
'''




#时间特征结构
#基础单层结构
class Layer1MLGAN_TF(tf.keras.Model):
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=1):
        super(Layer1MLGAN_TF,self).__init__()
        #GAN过程一直保持更新
        self.Textractor = TemporalFeatureMachine(spatialdim)
        #self.Sextractor = SpatialFeatureMachine(timedim)
        self.GANDenselayer = tf.keras.layers.Dense(1,name='GANDense')
        #GAN过程分层更新
        self.Downsampler1 = Downsampler(fiternums,1,'Downsampler1')#2-->12->24
        self.Upsampler1 = Upsampler(fiters=fiternums,rateindex=1,paddingdatalastshape=paddingshape,samplername='Upsampler1') #24-->12->1++>1            
    #特征合并
    
    def CombineFeature(self,tfeature,sfeature):
        tfeature = tf.expand_dims(tfeature,axis=-1)
        sfeature = tf.expand_dims(sfeature,axis=-1)
        comfeature= tf.concat([tfeature,sfeature],axis=-1)
        return comfeature
     
    def ExtractorFeature(self,mminp,adj_m):        
        #特征提取
        temporalf = self.Textractor(mminp)       
        #comf = self.CombineFeature(temporalf,temporalf)#特征合并
        comf = tf.expand_dims(temporalf,axis=-1)
        return comf

    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        upf1 = self.Upsampler1(downf1,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)           
        return res



class Layer2MLGAN_TF(Layer1MLGAN_TF):#2层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=24):
        super(Layer2MLGAN_TF,self).__init__(timedim,spatialdim,12,1)
        
        #GAN过程分层更新
        self.Downsampler2 = Downsampler(fiternums,2,'Downsampler2')#2 -->12->24 -->24->48
        self.Upsampler2 = Upsampler(fiters=fiternums,rateindex=2,paddingdatalastshape=24,samplername='Upsampler2')#48-->24->24++>24-->12->2 ++>2      
    def inherit(self, layer1gan):#继承权重参数
        self.Textractor = layer1gan.Textractor
        
        self.GANDenselayer = layer1gan.GANDenselayer      
        self.Downsampler1 = layer1gan.Downsampler1
        self.Upsampler1 = layer1gan.Upsampler1 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        upf2 = self.Upsampler2(downf2,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res
    
class Layer3MLGAN_TF(Layer2MLGAN_TF):#3层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=48):
        super(Layer3MLGAN_TF,self).__init__(timedim,spatialdim,12,24)        
        #GAN过程分层更新
        self.Downsampler3 = Downsampler(fiternums,3,'Downsampler3')#2 -->12->24-->24->48-->36->72
        self.Upsampler3 = Upsampler(fiters=fiternums,rateindex=3,paddingdatalastshape=48,samplername='Upsampler3')#72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer2gan):#继承权重参数
        self.Textractor = layer2gan.Textractor        
        self.GANDenselayer = layer2gan.GANDenselayer      
        self.Downsampler1 = layer2gan.Downsampler1
        self.Upsampler1 = layer2gan.Upsampler1 
        self.Downsampler2 = layer2gan.Downsampler2
        self.Upsampler2 = layer2gan.Upsampler2 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        upf3 = self.Upsampler3(downf3,downf2)
        upf2=self.Upsampler2(upf3 ,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res


class Layer4MLGAN_TF(Layer3MLGAN_TF):#4层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=72):
        super(Layer4MLGAN_TF,self).__init__(timedim,spatialdim,12,48)        
        #GAN过程分层更新
        self.Downsampler4 = Downsampler(fiternums,4,'Downsampler4')#2 -->12->24-->24->48-->36->72-->48->96
        self.Upsampler4 = Upsampler(fiters=fiternums,rateindex=4,paddingdatalastshape=72,samplername='Upsampler4')#96-->48->72++>72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer3gan):#继承权重参数
        self.Textractor = layer3gan.Textractor        
        self.GANDenselayer = layer3gan.GANDenselayer      
        self.Downsampler1 = layer3gan.Downsampler1
        self.Upsampler1 = layer3gan.Upsampler1 
        self.Downsampler2 = layer3gan.Downsampler2
        self.Upsampler2 = layer3gan.Upsampler2 
        self.Downsampler3 = layer3gan.Downsampler3
        self.Upsampler3 = layer3gan.Upsampler3 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        downf4 = self.Downsampler4(downf3)
        upf4 = self.Upsampler4(downf4,downf3)
        upf3 = self.Upsampler3(upf4,downf2)
        upf2=self.Upsampler2(upf3,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res

class Layer5MLGAN_TF(Layer4MLGAN_TF):#5层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=96):
        super(Layer5MLGAN_TF,self).__init__(timedim,spatialdim,12,72)        
        #GAN过程分层更新
        self.Downsampler5 = Downsampler(fiternums,5,'Downsampler4')#2 -->12->24-->24->48-->36->72-->48->96-->60->120
        self.Upsampler5 = Upsampler(fiters=fiternums,rateindex=5,paddingdatalastshape=96,samplername='Upsampler4')#120-->60->96++>96-->48->72++>72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer4gan):#继承权重参数
        self.Textractor = layer4gan.Textractor        
        self.GANDenselayer = layer4gan.GANDenselayer      
        self.Downsampler1 = layer4gan.Downsampler1
        self.Upsampler1 = layer4gan.Upsampler1 
        self.Downsampler2 = layer4gan.Downsampler2
        self.Upsampler2 = layer4gan.Upsampler2 
        self.Downsampler3 = layer4gan.Downsampler3
        self.Upsampler3 = layer4gan.Upsampler3 
        self.Downsampler4 = layer4gan.Downsampler4
        self.Upsampler4 = layer4gan.Upsampler4 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        downf4 = self.Downsampler4(downf3)
        downf5 = self.Downsampler5(downf4)
        upf5 = self.Upsampler5(downf5,downf4)
        upf4 = self.Upsampler4(upf5,downf3)
        upf3 = self.Upsampler3(upf4,downf2)
        upf2=self.Upsampler2(upf3,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res

#空间结构特征结构
#基础单层结构
class Layer1MLGAN_SF(tf.keras.Model):
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=1):
        super(Layer1MLGAN_SF,self).__init__()
        #GAN过程一直保持更新
        #self.Textractor = TemporalFeatureMachine(spatialdim)
        self.Sextractor = SpatialFeatureMachine(timedim)
        self.GANDenselayer = tf.keras.layers.Dense(1,name='GANDense')
        #GAN过程分层更新
        self.Downsampler1 = Downsampler(fiternums,1,'Downsampler1')#2-->12->24
        self.Upsampler1 = Upsampler(fiters=fiternums,rateindex=1,paddingdatalastshape=paddingshape,samplername='Upsampler1') #24-->12->2++>2            
    #特征合并
    
    def CombineFeature(self,tfeature,sfeature):
        tfeature = tf.expand_dims(tf.transpose(tfeature,perm=[0,2,1]),axis=-1)
        sfeature = tf.expand_dims(tf.transpose(sfeature,perm=[0,2,1]),axis=-1)
        comfeature= tf.concat([tfeature,sfeature],axis=-1)
        return comfeature
    #特征提取
        
    def ExtractorFeature(self,mminp,adj_m):        
        #特征提取
        spatialf = self.Sextractor([mminp,adj_m])
        comf  = tf.expand_dims(tf.transpose(spatialf,perm=[0,2,1]),axis=-1)
        #comf = self.CombineFeature(spatialf,spatialf)#特征合并
        
        return comf
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        upf1 = self.Upsampler1(downf1,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)           
        return res



class Layer2MLGAN_SF(Layer1MLGAN_SF):#2层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=24):
        super(Layer2MLGAN_SF,self).__init__(timedim,spatialdim,12,1)
        
        #GAN过程分层更新
        self.Downsampler2 = Downsampler(fiternums,2,'Downsampler2')#2 -->12->24 -->24->48
        self.Upsampler2 = Upsampler(fiters=fiternums,rateindex=2,paddingdatalastshape=24,samplername='Upsampler2')#48-->24->24++>24-->12->2 ++>2      
    def inherit(self, layer1gan):#继承权重参数
        #self.Textractor = layer1gan.Textractor
        self.Sextractor = layer1gan.Sextractor
        self.GANDenselayer = layer1gan.GANDenselayer      
        self.Downsampler1 = layer1gan.Downsampler1
        self.Upsampler1 = layer1gan.Upsampler1 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        upf2 = self.Upsampler2(downf2,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res
    
class Layer3MLGAN_SF(Layer2MLGAN_SF):#3层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=48):
        super(Layer3MLGAN_SF,self).__init__(timedim,spatialdim,12,24)        
        #GAN过程分层更新
        self.Downsampler3 = Downsampler(fiternums,3,'Downsampler3')#2 -->12->24-->24->48-->36->72
        self.Upsampler3 = Upsampler(fiters=fiternums,rateindex=3,paddingdatalastshape=48,samplername='Upsampler3')#72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer2gan):#继承权重参数
        self.Sextractor = layer2gan.Sextractor
        self.GANDenselayer = layer2gan.GANDenselayer      
        self.Downsampler1 = layer2gan.Downsampler1
        self.Upsampler1 = layer2gan.Upsampler1 
        
        self.Downsampler2 = layer2gan.Downsampler2
        self.Upsampler2 = layer2gan.Upsampler2 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        upf3 = self.Upsampler3(downf3,downf2)
        upf2=self.Upsampler2(upf3 ,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res


class Layer4MLGAN_SF(Layer3MLGAN_SF):#4层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=72):
        super(Layer4MLGAN_SF,self).__init__(timedim,spatialdim,12,48)        
        #GAN过程分层更新
        self.Downsampler4 = Downsampler(fiternums,4,'Downsampler4')#2 -->12->24-->24->48-->36->72-->48->96
        self.Upsampler4 = Upsampler(fiters=fiternums,rateindex=4,paddingdatalastshape=72,samplername='Upsampler4')#96-->48->72++>72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer3gan):#继承权重参数
        self.Sextractor = layer3gan.Sextractor
        self.GANDenselayer = layer3gan.GANDenselayer      
        self.Downsampler1 = layer3gan.Downsampler1
        self.Upsampler1 = layer3gan.Upsampler1 
        
        self.Downsampler2 = layer3gan.Downsampler2
        self.Upsampler2 = layer3gan.Upsampler2 
        self.Downsampler3 = layer3gan.Downsampler3
        self.Upsampler3 = layer3gan.Upsampler3 
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        downf4 = self.Downsampler4(downf3)
        upf4 = self.Upsampler4(downf4,downf3)
        upf3 = self.Upsampler3(upf4,downf2)
        upf2=self.Upsampler2(upf3,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res

class Layer5MLGAN_SF(Layer4MLGAN_SF):#5层结构
    def __init__(self,timedim,spatialdim,fiternums=12,paddingshape=96):
        super(Layer5MLGAN_SF,self).__init__(timedim,spatialdim,12,72)        
        #GAN过程分层更新
        self.Downsampler5 = Downsampler(fiternums,5,'Downsampler4')#2 -->12->24-->24->48-->36->72-->48->96-->60->120
        self.Upsampler5 = Upsampler(fiters=fiternums,rateindex=5,paddingdatalastshape=96,samplername='Upsampler4')#120-->60->96++>96-->48->72++>72-->36->48++>48-->24->24++>24-->12->2 ++>2            
    def inherit(self, layer4gan):#继承权重参数
        self.Sextractor = layer4gan.Sextractor
        self.GANDenselayer = layer4gan.GANDenselayer      
        self.Downsampler1 = layer4gan.Downsampler1
        self.Upsampler1 = layer4gan.Upsampler1 
        
        self.Downsampler2 = layer4gan.Downsampler2
        self.Upsampler2 = layer4gan.Upsampler2 
        self.Downsampler3 = layer4gan.Downsampler3
        self.Upsampler3 = layer4gan.Upsampler3 
        self.Downsampler4 = layer4gan.Downsampler4
        self.Upsampler4 = layer4gan.Upsampler4
    def call(self,inputs,adj_m):#inputs格式为(origindata,maskmatrix)
        #数据处理
        od,mm = inputs
        inp = od*mm       
        comf = self.ExtractorFeature(inp,adj_m)        
        #采样
        downf1 = self.Downsampler1(comf)
        downf2 = self.Downsampler2(downf1)
        downf3 = self.Downsampler3(downf2)
        downf4 = self.Downsampler4(downf3)
        downf5 = self.Downsampler5(downf4)
        upf5 = self.Upsampler5(downf5,downf4)
        upf4 = self.Upsampler4(upf5,downf3)
        upf3 = self.Upsampler3(upf4,downf2)
        upf2=self.Upsampler2(upf3,downf1)
        upf1 = self.Upsampler1(upf2,comf) 
        #gan维度调整
        #gan维度调整
        gandense= self.GANDenselayer(upf1)
        res = tf.reduce_sum(gandense,axis = -1)   
        return res