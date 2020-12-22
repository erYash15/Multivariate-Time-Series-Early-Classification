# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:52:08 2019

@author: hp
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as spio
from scipy import stats


def loaddatasetMTS(dname):
    path="./mtsdata/"+dname
    os.chdir(path)
    #path="./mtsdata/"+dname+"/"+dname+".mat"
    filename = dname+".mat"
    data = spio.loadmat(filename, squeeze_me=True)
    
    os.chdir("../../")
    
    print(os.getcwd())
    train = data['mts']['train'][()]
    trainlabels = data['mts']['trainlabels'][()]
    test = data['mts']['test'][()]
    testlabels = data['mts']['testlabels'][()]
    var = train[0].shape[0]
    # normalize train data
    for i in range(len(train)):
        for v in range(var):
            train[i]=train[i].astype('float64')
            if np.std(train[i][v])!=0:
                train[i][v] = stats.zscore(train[i][v])
    # normalize test data
    for i in range(len(test)):
        for v in range(var):
            test[i]=test[i].astype('float64')
            if np.std(test[i][v])!=0:
                test[i][v]= stats.zscore(test[i][v])
    return train,trainlabels,test,testlabels 

def loaddatasetUTS(dname):
    train_path="./UCR_TS_Archive_2015/"+dname+"/"+dname+"_TRAIN"
    test_path="./UCR_TS_Archive_2015/"+dname+"/"+dname+"_TEST"
    train=pd.read_csv(train_path, header=None)
    test=pd.read_csv(test_path, header=None)
    Xtrain = train.values
    Xtest = test.values
    
#    train = Xtrain[:,1:]
#    trainlabels = Xtrain[:,0]
#    test =Xtest[:,1:]
#    testlabels = Xtest[:,0] 
#    return train,trainlabels,test,testlabels 
    return Xtrain,Xtest
   
# main code
########### load multivariate dataset###############

#train,trainlabels,test,testlabels = loaddatasetMTS('Wafer')

############## load univariate dataset  ##################
train, test = loaddatasetUTS('CBF')
#candidates = generate_candidates(data,50,5)
shapelet_dict = extract_shapelets(train)
