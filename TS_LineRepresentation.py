'''reading the MTS data'''
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
	
'''Actual Ploting Code'''
X = [i for i in range(200)]
plt.figure(figsize=(50,25))
for i in range(len(train)):
    if(trainlabels[i]==1):
        print("",end=",")
        plt.plot(train[i][1], color='g')
        plt.plot(train[i][0], color='b')
    else:
        print("",end=",")
        plt.plot(train[i][1], color='g')
        plt.plot(train[i][0], color='b')
plt.show()

'''Note - very informative but hard to read on large datasets.'''