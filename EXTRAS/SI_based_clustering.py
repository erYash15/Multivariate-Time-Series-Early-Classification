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
    
    
def dtw_dist(X,Y):
    return dtw(X,Y)


new_sample = []
for k in range(10):
    temp = [ float(random.uniform(1,2)) for i in range(10)]
    new_sample.append(temp)
for k in range(10):
    temp = [ float(random.uniform(50,60)) for i in range(10)]
    new_sample.append(temp)
for k in range(10):
    temp = [ float(random.uniform(100,120)) for i in range(10)]
    new_sample.append(temp)
    

import matplotlib.pyplot as plt
plt.plot(np.asarray(new_sample).T)

def SI_clust(data,no_cluster,num_iters):

    data_idxs = [[int(i), random.randint(0, no_cluster-1)] for i in range(len(data))]
    pred_labels = [ g[1] for g in data_idxs ]
    print("initalize classs", pred_labels)
    #pred_labels = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0]
    dist = [ [ 0 for i in range(len(data)) ] for j in range(len(data))]
    #filling distance matrix
    for i in range((len(data))):
        for j in range(len(data)):
            if(i==j):
                continue
            elif(dist[j][i]==0):
                dist[i][j] = min_euclidean(data[i],data[j])
                dist[j][i] = dist[i][j]
    for mmm in data:
        print(mmm)
    
    
    s_label = 0
    SI_avg = []
    
    for _ in range(num_iters):
        count = 0
        SI = []
        closest_cluster = []
        for i in range(len(data)):
            a = 0
            b = float("inf")
            #find distance from each cluster
            c_d = [int(0) for k in range(no_cluster)]
            for k in range(len(dist[i])):
                c_d[pred_labels[k]] += dist[i][k]
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
        print("old labels\n\n",pred_labels)
        print("SI\n\n", SI)
        print("Closest\n\n", closest_cluster)
        count_neg = 0
        if(SI.count(-1.0)==len(data)):
            pred_labels = [random.randint(0, no_cluster-1) for ll in range(len(data))]
            continue
        for kk in range(len(data)):
            if(SI[kk] < 0):
                pred_labels[kk] = closest_cluster[kk]
                count_neg += 1
        print(SI)
        SI_avg.append(mean(SI))
        
        if(count_neg == 0):
            break
        #goes for another iteration
    plt.plot(np.asarray(SI_avg).T)
    return pred_labels

labels = SI_clust(new_sample,3,5)
print(labels)