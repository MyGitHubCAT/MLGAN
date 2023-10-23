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


class ENV(object):
    def __init__(self):
        #data dirpath
        self.codepath = os.getcwd()
        self.path = os.path.join(self.codepath,'PEMS_2013_D5_data','recdir')
        self.adj_csvpath =os.path.join(self.codepath,'PEMS_2013_D5_data','PEMS_VDS_data_resort.csv')
        self.rmdir =os.path.join(self.codepath,'PEMS_2013_D5_data','pydatafile')
        
        self.timescale = 288
        self.spatialscale = 0
        self.missingrate = np.arange(19)#index from 0 to 18   missingrate from 95% to 5%
        self.RandNUMS = 10
        
        self.learning_rate=1e-4
        self.batch_size = 12
        
        #gan
        self.GANtrainmetrics=tf.keras.metrics.RootMeanSquaredError(name='testRMSE')
        self.MLGAN_genoptimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.MLGAN_discoptimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

        #cor
        self.corloss = tf.keras.losses.MeanSquaredError()
        self.coropt = tf.keras.optimizers.Adam(self.learning_rate)
        self.trainmetrics=tf.keras.metrics.RootMeanSquaredError(name='testRMSE')
        
        #train
        self.gan_epochs = 30
        self.cor_epochs = 50
        
    def setadj(self,adj):
        self. ADJMatrix = adj  

env = ENV()


def convertdataset_workday_weekend(dirpath):
    files= os.listdir(dirpath)
    datafilepath= [os.path.join(dirpath,i) for i in files]#readCSV from dirpath
    datadfs = [pd.read_csv(i) for i in datafilepath]#read data
    
    #select flow data and merge data from different sensors 
    flowarrlist = [np.array(i['Flow (Veh/5 Minutes)']) for i in datadfs]
    flowarrs = np.stack(flowarrlist,axis = 1)
    
    
    #define the time and space dimension
    env.spatialscale = flowarrs.shape[1]#spatial space
    #time space
    env.timescale=288 # we divide a day into 288 piece(each piece for 5 minutes)
    TimeNums = int(flowarrs.shape[0]/env.timescale)
    
    #Sort by day
    st_dataset=[]
    for i in range(TimeNums):
        st_data = flowarrs[i*env.timescale:(i+1)*env.timescale,:].astype('float32')    
        st_dataset.append(st_data)
    st_dataset = np.array(st_dataset)
    st_dataset=st_dataset.astype('float32')
    
    #normalization
    datamean = st_dataset.mean()
    datastd =st_dataset.std()

    st_dataset_normal = (st_dataset-datamean)/datastd
    
    return st_dataset_normal,datamean,datastd


def split_workdayandweekend(st_dataset_normal):#Separate weekend and weekday data base on the dataset(Need to be customized if using your own dataset)
    dataset = st_dataset_normal
    wrkd_dataset=[]
    wekd_dataset=[]

    days_in_week = 7

    wrkd_dataset.append(dataset[0])
    wrkd_dataset.append(dataset[1])
    wrkd_dataset.append(dataset[2])
    wrkd_dataset.append(dataset[3])
    wekd_dataset.append(dataset[4])
    wekd_dataset.append(dataset[5])

    for i in range(21):
        wrkd_dataset.append(dataset[6+7*i])
        wrkd_dataset.append(dataset[6+7*i+1])   
        wrkd_dataset.append(dataset[6+7*i+2])
        wrkd_dataset.append(dataset[6+7*i+3])
        wrkd_dataset.append(dataset[6+7*i+4])
    
        wekd_dataset.append(dataset[6+7*i+5])
        wekd_dataset.append(dataset[6+7*i+6])
    
    wrkd_dataset=np.array(wrkd_dataset)
    wekd_dataset=np.array(wekd_dataset)
    return wrkd_dataset,wekd_dataset


#we record the adjacency of the sensors
#the definition of the adjacency matrix can be given based on different correlation
#herer we mainly considered the connecticity
def giveadj(adj_csvpath):
    
    adjdf = pd.read_csv(adj_csvpath) 

    ADJMatrix = np.zeros(shape=(len(adjdf),len(adjdf)))

    adjsum = 0

    adjvds = np.array(adjdf['adj vds'])
    for i in range(len(adjvds)):
        adji = adjvds[i].split('%')
        adjsum = adjsum + len(adji)
        for j in adji:       
            adjindex = adjdf[adjdf['vds']==int(j)].index.tolist()[0]       
            ADJMatrix[i][adjindex] = 1
        
    adjsum = adjsum + len(adjdf)#total adjacency value
    EMatrix = np.eye(len(adjdf))#unit matrix

    
    ADJMatrix =  ADJMatrix + EMatrix

    if(ADJMatrix.sum() != adjsum):
        print('adjacency matrix error')
        return null
    return ADJMatrix


def Maskmatrixes(rmdir):
    rmfiles = os.listdir(env.rmdir)
    rmfiles
    rminfiles = [np.fromfile(os.path.join(rmdir,i),dtype = np.int32) for i in rmfiles]
    for i in rminfiles:
        i.shape = ( env.timescale,env.spatialscale)


    #ten random int(0-20) matrix has been applied.
    #based the defined random matrix, mask matrix samples can be set as the missing rate increase
    maskmatrixes = []
    for i in env.missingrate:
        for j in rminfiles:
            maskmatrixes.append(np.where(j>i,1,0))#there is 380 mask matrix samples.
    maskmatrixes = np.array(maskmatrixes,dtype='float32')
    return maskmatrixes 


def discriminator_loss(real_output, fake_output,miss_output):#判别器损失
    real_loss = env.cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = env.cross_entropy(tf.zeros_like(fake_output), fake_output)
            
    #miss_loss = cross_entropy(tf.zeros_like(miss_output), miss_output)
    #total_loss = 0.4*real_loss + 0.4*fake_loss + 0.2*miss_loss
    total_loss =real_loss +fake_loss
    return total_loss

def generator_loss(fake_output):
    return env.cross_entropy(tf.ones_like(fake_output), fake_output)



#train for gan
def train_step(gen,disc,inputs,adj):
    traindata,maskmatrix = inputs    
    missdata = traindata*maskmatrix    
    with tf.GradientTape() as tapeforgan,tf.GradientTape() as tapefordisc:        
        gen_data = gen(inputs,adj)       
        real_output = disc(traindata)        
        fake_output = disc(gen_data)        
        miss_output = disc(missdata)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output,fake_output, miss_output)
        
        gradients_for_gen = tapeforgan.gradient(gen_loss,gen.trainable_variables)
        gradients_for_disc = tapefordisc.gradient(disc_loss,disc.trainable_variables)
        
        env.MLGAN_genoptimizer.apply_gradients(zip(gradients_for_gen,gen.trainable_variables))
        env.MLGAN_discoptimizer.apply_gradients(zip(gradients_for_disc,disc.trainable_variables))        
def train(gen,disc,dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for data_batch in dataset:
            traindata,maskmatrix = data_batch
            train_step(gen,disc,data_batch,env.ADJMatrix)
            env.GANtrainmetrics.update_state(gen(data_batch,env.ADJMatrix),traindata)
            #print(time.time()-start)
           # display.clear_output(wait=True)    
        print ('Time for epoch {} is {} sec  testRMSE:{:.4f}'.format(epoch + 1, time.time()-start,env.GANtrainmetrics.result()))
        env.GANtrainmetrics.reset_states()

#train correctionlayer
def train_step_forcorrection(MLGAN,inps,adjm):   
    with tf.GradientTape() as MLGAN_tape:
        label,mm = inps
        labelpart = label#*(1-mm)
        cordata = MLGAN(inps,adjm)
        corpart = cordata#*(1-mm)
        mlganloss = env.corloss(corpart,labelpart)
    MLGANgradients = MLGAN_tape.gradient(mlganloss,MLGAN.trainable_variables)
    env.coropt.apply_gradients(zip(MLGANgradients,MLGAN.trainable_variables))
def trainMLGAN(MLGANmodel,dataset,epochs):
    for i in range(epochs):
        start = time.time()
        for inps in dataset:           
            train_step_forcorrection(MLGANmodel,inps,env.ADJMatrix)
            
            x,mm = inps
            corx= x*(1-mm)
            env.trainmetrics.update_state(MLGANmodel(inps,env.ADJMatrix)*(1-mm),corx)
            
        print ('Time for epoch {} is {} sec  testRMSE:{:.4f}'.format(i + 1, time.time()-start,env.trainmetrics.result()))
        env.trainmetrics.reset_states()


def impute_data(inp,mask):
    onestf = tf.ones_like(mask).numpy()
    
    maskrate = mask.numpy().sum()/onestf.sum()
    missingrate = 1 - maskrate
    
    if missingrate <= 0.2:
        print('missrate:{},MLGAN with layer1'.format(missingrate))
        inps = inp,mask
        cordata = MLGANcorL1M5(inps,ADJMatrix)
        return cordata
    if missingrate <= 0.4:
        print('missrate:{},MLGAN with layer2'.format(missingrate))
        inps = inp,mask
        cordata = MLGANcorL2M5(inps,ADJMatrix)
        return cordata
    if missingrate <= 0.6:
        print('missrate:{},MLGAN with layer3'.format(missingrate))
        inps = inp,mask
        cordata = MLGANcorL3M5(inps,ADJMatrix)
        return cordata
    if missingrate <= 0.8:
        print('missrate:{},MLGAN with layer4'.format(missingrate))
        inps = inp,mask
        cordata = MLGANcorL4M5(inps,ADJMatrix)
        return cordata
    else:
        print('missrate:{},MLGAN with layer5'.format(missingrate))
        inps = inp,mask
        cordata = MLGANcorL5M5(inps,ADJMatrix)
        return cordata


def CaltotalMAE(origdata,compdata,nums,tdims,sdims):
    distance=0
    for i in range(nums):
        for j in range(tdims):
            for k in range(sdims):
                distance = distance + abs(origdata[i][j][k] - compdata[i][j][k])
    MAEvalue = distance/(nums*tdims*sdims)
    return MAEvalue

def CaltotalRMSE(origdata,compdata,nums,tdims,sdims):
    distance=0
    for i in range(nums):
        for j in range(tdims):
            for k in range(sdims):
                distance = distance + math.pow((origdata[i][j][k] - compdata[i][j][k]),2)
    RMSEvalue = math.sqrt(distance/(nums*tdims*sdims))
    return RMSEvalue

def showloss(oridata,impdata,mm):
    impmae = CaltotalMAE(oridata,impdata,oridata.shape[0],oridata.shape[1],oridata.shape[2])
    imprmse = CaltotalRMSE(oridata,impdata,oridata.shape[0],oridata.shape[1],oridata.shape[2])
    
    print('IMPMAE :'+ str(impmae))
    print('IMPRMSE :'+ str(imprmse))
   
    #return [impmae,imprmse,impr2,impmape]

def returnloss(oridata,impdata,mm):
    impmae = CaltotalMAE(oridata,impdata,oridata .shape[0],oridata.shape[1],oridata.shape[2])
    imprmse = CaltotalRMSE(oridata ,impdata,oridata.shape[0],oridata.shape[1],oridata.shape[2])
    
    #print('IMPMAE :'+ str(impmae))
    #print('IMPRMSE :'+ str(imprmse))
    #print('IMPR2 :'+ str(impr2))
    #print('IMPMAPE :'+ str(impmape))
    return [round(impmae,4),round(imprmse,4)]