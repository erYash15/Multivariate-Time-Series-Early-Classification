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
from tslearn.metrics import dtw
import random
from statistics import mean 

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



def dtw_distance(X,Y):
    return dtw(X,Y)

def euclidian(X,Y):
    return np.linalg.norm(X-Y)

def min_euclidean(X,Y):
    # X=Mx1, Y=Nx1
    min = float("inf")
    if(len(X)>len(Y)):
        for i in range(len(X)-len(Y)+1):
            dist=euclidian(np.array(X[i:i+len(Y)]),np.array(Y))
            if(dist<min):
                min = dist
        return(min)
    else:
        for i in range(len(Y)-len(X)+1):
            dist=euclidian(np.array(Y[i:i+len(X)]),np.array(X))
            if(dist<min):
                min = dist
        return(min)

'''useful function from selction'''
def MIL(shapelet_local,X):
    #shapelet_local[0] = shapelet
    #shapelet_local[2] is threshold
    for i in range(len(X)-len(shapelet_local[0])+1):
        dist=np.linalg.norm(np.array(X[i:i+len(shapelet_local[0])]-np.array(shapelet_local[0])))
        if(dist<shapelet_local[2]):
            return i+len(shapelet_local[0])
    return "NAN"
#returns the MIL distance if found else total length

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

        
def SI_clust(data,no_cluster,num_iters):
    data_idxs = [[int(i), random.randint(0, no_cluster-1)] for i in range(len(data))]
    pred_labels = [ g[1] for g in data_idxs ]
    s_label = 0
    for _ in range(num_iters):
        SI = []
        closest_cluster = []
        for i in range(len(data)):
            dist = []
            for s in range(len(data)):#s iterate for all data
                dist.append(min_euclidean(data[i],data[s]))
            a = 0
            b = float("inf")
            #find distance from each cluster
            c_d = [int(0) for k in range(no_cluster)]
            for k in range(len(dist)):
                c_d[pred_labels[k]] += dist[k]
            #normalize the distance
            for k in range(no_cluster):
                if(k == pred_labels[i]):
                    norm = pred_labels.count(k)
                    if(norm!=0):
                        c_d[k] /= (norm)
                else:    
                    norm = pred_labels.count(k)
                    if(norm<2):
                        norm = 2
                    c_d[k] /= (norm-1)
            
            #finding closest cluster
            for k in range(len(c_d)):
                if(k == pred_labels[i]):
                    a = c_d[k]
                else:
                    old_b = b
                    b = min(b,c_d[k])
                    if(old_b != b):
                        s_label = k
            
            #saving closest cluster for each point
            closest_cluster.append(s_label)
            #calculating si's
            si = (b-a)/max(b,a)
            SI.append(si)
        
        count_neg = 0
        if(SI.count(-1.0)==len(data)):
            pred_labels = [random.randint(0, no_cluster-1) for ll in range(len(data))]
            continue
        for kk in range(len(data)):
            if(SI[kk] < 0):
                pred_labels[kk] = closest_cluster[kk]
                count_neg += 1
        if(count_neg == 0):
            break
        #goes for another iteration
    #print(pred_labels)
    return pred_labels,mean(SI),SI





if __name__ == '__main__':
    train,trainlabels,test,testlabels = loaddatasetMTS("ECG")
    
    with open('output_fss_sts', 'rb') as f:
        shapelets = pickle.load(f) 
    print(len(shapelets))
    
    core_feature_si = []
    
    uniq_lab = set(trainlabels)
    uniq_var = len(train[0])
    data = [ [[] for j in range(uniq_var)] for i in range(len(uniq_lab))]
    print(data)
    for i in shapelets:
        data[i[1]-1][i[5]].append(i)
        
    print(len(data[0][0]))
    print(len(data[0][1]))
    print(len(data[1][0]))
    print(len(data[1][1]))
    
    core_features = []
    
    for cc in range(0,len(data)):
        for vv in range(0,len(data)):
            new_labels = []
            new_si_score = 0
            new_si_sample = []
            print(cc,vv)
            temp = [d[0] for d in data[cc][vv]]
            
            for k in range(3,4):
                print("k",k)
                labels,si_score,si_sample = SI_clust(temp,k,10)
                print(si_score)
                if (new_si_score < si_score):
                    new_si_score = si_score
                    new_labels = labels
                    new_si_sample = si_sample
            
            imp_features = [ [] for q in range(len(set(new_labels)))]
            
            GEFM_lst = [[y for x in range(2)] for y in range(len(data[cc][vv]))]
            counter = 0
            for shapelet_local in data[cc][vv]:#for each shapelet in all shaplets
                print(counter)
                earliness = earli_cal(shapelet_local,train)
                #shapelet_local[3] is precision
                #shapelet_local[4] is recall
                if(earliness == 0 or shapelet_local[3]==0 or shapelet_local[4]==0):
                    GEFM=0
                else:
                    GEFM=1/((1/earliness)+(1/shapelet_local[3])+(1/shapelet_local[4]))
                GEFM_lst[counter][0] = GEFM
                counter = counter + 1
            #print(shapelets_local)
            GEFM_lst.sort(reverse= True)
            
            for imp in range(len(imp_features)):
                if(len(imp_features[imp])==0):
                    for g in GEFM_lst:
                        if(new_labels[g[1]]==imp):
                            imp_features[imp] = data[cc][vv][g[1]]
                            break
            print(imp_features)
            core_features.extend(imp_features)
    print(core_features)
    with open('core_features_fss_sts', 'wb') as f:
            pickle.dump(core_features, f)