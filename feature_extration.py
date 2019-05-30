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


#####MODULE-3######
'''pre - processing data'''
train = np.concatenate((train, test), axis=0)
trainlabels = np.concatenate((trainlabels, testlabels), axis=0)

train, test, trainlabels, testlabels = train_test_split(train, trainlabels, test_size=0.10)
'''saving processed files'''
with open('train', 'wb') as f:
    pickle.dump(train, f) 
with open('test', 'wb') as f:
    pickle.dump(test, f) 
with open('trainlabels', 'wb') as f:
    pickle.dump(trainlabels, f) 
with open('testlabels', 'wb') as f:
    pickle.dump(testlabels, f)
'''feature extraction'''
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
	
def Learnthreshold(candidate,data,labels,ss,tt):
    # candidate: Px1, data: MxVxT, labels: Mx1, s: integer value, t:integer value  
    dist_mtx = []
    thre_cand = []
    sorted_labels = []
    total_c = np.count_nonzero(labels == labels[ss])
    for s in range(data.shape[0]):
        dist_mtx.append([min_euclidean(data[s][tt],candidate),labels[s]])
    dist_mtx.sort()
    for i in dist_mtx:
        sorted_labels.append(i[1])
    for i in range(len(dist_mtx)-1):
        threshold = (dist_mtx[i][0]+dist_mtx[i+1][0])/2
        precision = (sorted_labels[0:i+1].count(labels[ss]))/len(dist_mtx[0:i+1])
        recall = (sorted_labels[0:i+1].count(labels[ss]))/total_c
        if(precision==0 or recall==0):
            quality = 0
        else:
            quality = 2/((1/precision)+(1/recall))
        thre_cand.append([threshold,precision,recall,quality])
    max_qual = 0
    for i in range(len(thre_cand)):   
        if((thre_cand[i][3]) > max_qual):
            max_qual = thre_cand[i][3]
            index = i
    return thre_cand[index]

def feature_extraction(data,labels,Lmin=5,Pre_min=0.51,Rec_min=0.10):
    output = []
    for s in range(data.shape[0]):
        print(s,end=",")
        for t in range(data[s].shape[0]):
            Lmax = len(data[s][t])//3
            for L in range(Lmin,Lmax+1):  # example min=4 and max=7, range(min,max+1)={4,5,6,7}
                for i in range(0,len(data[s][t])-L+1):
                    candidate = data[s][t][i:i+L]
                    threshold_info = Learnthreshold(candidate,data,labels,s,t)
                    if(threshold_info[1]>Pre_min and threshold_info[2]>Rec_min):
                        output.append([candidate,labels[s],threshold_info[0],threshold_info[1],threshold_info[2],t,L,s])
    return output
	
shapelets = feature_extraction(train,trainlabels)

'''Saving shapelet file as output'''
with open('output', 'wb') as f:
    pickle.dump(shapelets, f) 
