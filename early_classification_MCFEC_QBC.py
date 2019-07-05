'''old imports'''
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
import random
import statistics 


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


with open('core_features_greedy', 'rb') as f:
    core_features = pickle.load(f)


def max_feq(X):
    try:
        res = statistics.mode(X)
    except:
        res = 0
    return res

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

for c in core_features:
    print(c[1],c[5])
    
'''QBC TECHNIQUE'''	
    
pred_labels = []
earliness = 0
early = []
uniq_var = len(train[0])
uniq_lab = set(trainlabels)
temp = []

for s in range(test.shape[0]):
    temp = []
    for j in range(uniq_var):
        temp.append([len(test[s][0]),len(test[s][0])])
    early.append(temp)

for s in range(test.shape[0]):
    pred = []
    for _ in range(uniq_var):
        pred.append([])
    for t in range(test[s].shape[0]):
        for L in range(1,len(test[s][t])):
            temp = test[s][t][0:L]
            for c in range(len(core_features)):
                if(len(core_features[c][0])<L and core_features[c][5]==t):
                   # print(min_euclidean(temp,core_features[c][0]) , core_features[c][2])
                    if(min_euclidean(temp,core_features[c][0]) < core_features[c][2]):
                        if(early[s][t][0]==early[s][t][1]):
                            early[s][t][0] = L
                            pred[t].append(core_features[c][1])
    flat_pred = [item for sublist in pred for item in sublist]
    pred_labels.append(flat_pred)

print(pred_labels)

'''calculating earliness'''
earliness = 0
count = 0
for i in range(len(early)):
    L = 0
    full_L = 0
    for k in range(uniq_var):
        L = max(L,early[i][k][0])
        full_L=max(full_L,early[i][k][1])
    earliness += 1 - max(early[i][0][0],early[i][1][0])/max(early[i][0][1],early[i][1][1])

print("earliness on data",earliness/100)

'''predicting labels'''
pred_labels_test = []
count = 0
for p in pred_labels:
    print(p)
    if(max_feq(p)==0):
        pred_labels_test.append(int(random.randint(1,len(set(trainlabels)))))  
    else:
        pred_labels_test.append(max_feq(p))   
        #if(early[count][0][0]>early[count][1][0]):
        #    pred_labels_test.append(p[0][0])
        #else:
        #    pred_labels_test.append(p[1][0]) 
        count += 1      
      
print(pred_labels_test)
		 
'''finding accuracy'''		 
count = 0
for i in range(len(test)):
    #print(pred_labels_test[i],pred_labels[i])
    if(pred_labels_test[i]==testlabels[i]):
        count += 1
print("accuracy is ", count/len(test))