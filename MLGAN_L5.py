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

#example MLGAN(5 layers)
class MLGAN(tf.keras.Model):
    def __init__(self,mlganl1,mlganl2,mlganl3,mlganl4,mlganl5):
        super(MLGAN,self).__init__()
        self.mlganl1 = mlganl1
        self.mlganl2 = mlganl2
        self.mlganl3 = mlganl3
        self.mlganl4 = mlganl4
        self.mlganl5 = mlganl5
    
        self.mlganl1.trainable = False
        self.mlganl2.trainable = False
        self.mlganl3.trainable = False
        self.mlganl4.trainable = False
        self.mlganl5.trainable = False
        
        self.ADJMatrix = None
        self.activegan = None

    def call(self,inputs,adj_m):
        inp,mask =inputs
        self.ADJMatrix =adj_m
        cordata = self.impute_Data(inputs)
        resdata = cordata*(1-mask)+ inp*mask 
        return resdata 
    
    def impute_Data(self,inps):
        _,mask = inps
        onestf = tf.ones_like(mask).numpy()

        maskrate = mask.numpy().sum()/onestf.sum()
        missingrate = 1 - maskrate
        print()
        if missingrate <= 0.2:
            print('missrate:{},MLGAN with layer1'.format(missingrate))
            
            cordata = self.mlganl1(inps,self.ADJMatrix)
            self.activegan = self.mlganl1.gan
            return cordata
        if missingrate <= 0.4:
            print('missrate:{},MLGAN with layer2'.format(missingrate))
            
            cordata = self.mlganl2(inps,self.ADJMatrix)
            self.activegan = self.mlganl2.gan
            return cordata
        if missingrate <= 0.6:
            print('missrate:{},MLGAN with layer3'.format(missingrate))
            
            cordata = self.mlganl3(inps,self.ADJMatrix)
            self.activegan = self.mlganl3.gan
            return cordata
        if missingrate <= 0.75:
            print('missrate:{},MLGAN with layer4'.format(missingrate))
            
            cordata = self.mlganl4(inps,self.ADJMatrix)
            self.activegan = self.mlganl4.gan
            return cordata
        else:
            print('missrate:{},MLGAN with layer5'.format(missingrate))
            cordata = self.mlganl5(inps,self.ADJMatrix)
            self.activegan = self.mlganl5.gan
            return cordata