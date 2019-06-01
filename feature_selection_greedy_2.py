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






#'''loading all shapelets, quality,datasets '''
#with open('features', 'rb') as f:
#    features = pickle.load(f)
#with open('output', 'rb') as f:
#    shapelets = pickle.load(f)
#with open('train', 'rb') as f:
#    train = pickle.load(f)
#with open('trainlabels', 'rb') as f:
#    trainlabels = pickle.load(f)

shapelets_local = copy.deepcopy(shapelets)

def train_shapelets(shapelet_local,Y):
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
    data_cpy = copy.deepcopy(data)
    labels_cpy = copy.deepcopy(labels)
    imp_shapelets = []
    for g in features:
        print("g no. is",g)
        flag = 1;counter = 0
        prev_len = len(data_cpy)
        while(len(data_cpy)!=0 and flag==1):
            pred = train_shapelets(shapelets_local[g[1]],data_cpy[counter][shapelets_local[g[1]][5]])
            if(pred == "NAN" or pred != labels_cpy[counter]):
                counter = counter + 1
            else:
                data_cpy = np.delete(data_cpy,counter)
                labels_cpy = np.delete(labels_cpy,counter)
                print("length of data",len(data_cpy),"length of labels",len(labels_cpy))
            if(counter>=len(data_cpy)):
                print("counter is",counter)
                flag = 0
        curr_len = len(data_cpy)
        print("diff", prev_len - curr_len)
        if(prev_len>curr_len):
            print("selected g is ", g[1])
            imp_shapelets.append(shapelets_local[g[1]])
    return imp_shapelets
	
	
core_feature = imp_features(shapelets_local,train,trainlabels,features)
print(len(core_feature))
with open('core_features', 'wb') as f:
    pickle.dump(core_feature,f)