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



###MODULE-2###
'''useful function from selction'''
def MIL(shapelet_local,X):
    #shapelet_local[0] = shapelet
    #shapelet_local[2] is threshold
    for i in range(len(X)-len(shapelet_local[0])+1):
        dist=np.linalg.norm(np.array(X[i:i+len(shapelet_local[0])]-np.array(shapelet_local[0])))
        if(dist<shapelet_local[2]):
            return i+len(shapelet_local[0])
    return "NAN"
#returns the MIL distance if fouund else total length
	
def earli_cal(shapelet_local,data):
    earliness = 0
    count = 0
    for s in range(data.shape[0]):
        #shapelet_local[5] is variate
        mil_length = MIL(shapelet_local,data[s][int(shapelet_local[5])])
        if(mil_length != "NAN"):
            early = mil_length/len(data[s][int(shapelet_local[5])])
            earliness += (1 - early)
            count += 1
    earliness = earliness/count
    return earliness


###MODULE-3###
'''fuction of finding quality'''
def feature_selection(shapelets_local,data,labels,w0=1,w1=1,w2=1):
    #shapelets_local is list of list of shapelets with information - shapelets,class,threshold,prec,recall,variate,length,sample
    
    GEFM_lst = [[y for x in range(2)] for y in range(len(shapelets_local))]
    counter = 0
    for shapelet_local in shapelets_local:#for each shapelet in all shaplets
        print(counter)
        earliness = earli_cal(shapelet_local,data)
        #shapelet_local[3] is precision
        #shapelet_local[4] is recall
        if(earliness == 0 or shapelet_local[3]==0 or shapelet_local[4]==0):
            GEFM=0
        else:
            GEFM=1/((w0/earliness)+(w1/shapelet_local[3])+(w2/shapelet_local[4]))
        GEFM_lst[counter][0] = GEFM
        counter = counter + 1
    #print(shapelets_local)
    GEFM_lst.sort(reverse= True)
    print(GEFM_lst)
    return GEFM_lst


# Main Code Start
if __name__ == '__main__':
    train,trainlabels,test,testlabels = loaddatasetMTS("ECG")
    ###MODULE-1###
    '''reading all shapelets from data'''
    with open('output_fss_sts', 'rb') as f:
        shapelets = pickle.load(f)
    shapelets_local = copy.deepcopy(shapelets)	
    features = feature_selection(shapelets_local,train,trainlabels)
    '''saving all shapelets quality'''
    with open('features_greedy_fss_sts', 'wb') as f:
        pickle.dump(features,f)
