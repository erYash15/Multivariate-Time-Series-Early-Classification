'''old code'''
#####MODULE-1######
'''importing libraries '''
import itertools
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as spio
from scipy import stats
from sklearn.model_selection import train_test_split
import csv
import pickle

#####MODULE-2######
'''loading .mat file data set of MTS'''
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
    #var = train[0].shape[0]
    # normalize train data
    #for i in range(len(train)):
    #    for v in range(var):
    #        train[i]=train[i].astype('float64')
    #        if np.std(train[i][v])!=0:
    #            train[i][v] = stats.zscore(train[i][v])
    # normalize test data
    #for i in range(len(test)):
    #    for v in range(var):
    #        test[i]=test[i].astype('float64')
    #        if np.std(test[i][v])!=0:
    #            test[i][v]= stats.zscore(test[i][v])
    return train,trainlabels,test,testlabels

train,trainlabels,test,testlabels = loaddatasetMTS("ECG")

def euclidian(X,Y):
    return np.linalg.norm(X-Y)
	
def min_euclidean(X,Y):
    # X=Mx1, Y=Nx1
    min = float("inf")
    for i in range(len(X)-len(Y)+1):
        dist=euclidian(np.array(X[i:i+len(Y)]),np.array(Y))
        if(dist<min):
            min = dist
    return(min)


def train_shapelets_pred(shapelet_local,Y):
    #print(shapelet_local[0],Y)
    if(len(shapelet_local[0])<len(Y)):
        for i in range((len(Y)-len(shapelet_local[0]))+1):
            dist=euclidian(np.array(Y[i:i+len(shapelet_local[0])]),np.array(shapelet_local[0]))
            if(dist<=shapelet_local[2]):
                return shapelet_local[1]
        return "NAN"
    else:
        for i in range((len(shapelet_local[0]-len(Y)))+1):
            dist=euclidian(np.array(shapelet_local[i:i+len(Y)]),np.array(Y))
            if(dist<=shapelet_local[2]):
                return shapelet_local[1]
        return "NAN"
		
with open('output', 'rb') as f:
    shapelets = pickle.load(f)

with open('features', 'rb') as f:
    features = pickle.load(f)
		

    
def imp_features(shapelets_local,data,labels,features):
    
    imp_shapelets = []
    uniq_lab = set(labels)
    uniq_var = len(data[0])
    all_data_variatewise = []
    for _ in range(uniq_var):
        all_data_variatewise.append([i for i in range(len(data))])
    
    all_shapelets_variatewise = []
    for _ in range(uniq_var):
        all_shapelets_variatewise.append([])
    
    for g in features:
        all_shapelets_variatewise[shapelets_local[g[1]][5]].append(g[1])

    
    for variate in range(len(all_shapelets_variatewise)):
        for shapelets in all_shapelets_variatewise[variate]:
            #print(shapelets)
            prev_len = len(all_data_variatewise[variate])
            temp = []
            for i in all_data_variatewise[variate]:
                print(shapelets,i,variate)
                pred = train_shapelets_pred(shapelets_local[shapelets],data[i][variate]) 
                if(pred == labels[i]):
                    temp.append(i)
            for j in temp:
               all_data_variatewise[variate].remove(j)
            curr_len = len(all_data_variatewise[variate])
            if(prev_len>curr_len):
                imp_shapelets.append(shapelets_local[shapelets])
                print(imp_shapelets)
   
    return imp_shapelets
    
	
core_feature = imp_features(shapelets,train,trainlabels,features)
print(len(core_feature))
with open('core_features', 'wb') as f:
    pickle.dump(core_feature,f)
