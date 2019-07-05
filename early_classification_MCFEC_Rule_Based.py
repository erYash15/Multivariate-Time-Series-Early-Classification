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
from sklearn.metrics import f1_score, precision_score, recall_score
import random
from itertools import combinations

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

def euclidian(X,Y):
    return np.linalg.norm(X-Y)

def min_euclidean(X,Y):
    min = float("inf")
    for i in range(len(X)-len(Y)+1):
        dist=euclidian(np.array(X[i:i+len(Y)]),np.array(Y))
        if(dist<min):
            min = dist
    return(min)

def prec(X,Y,lab):
    #X is list of true labels
    #Y is list of predicted labels
    Y_lab = []
    X_lab = []
    for i in range(len(X)):
        if(X[i]==lab):
            X_lab.append(i)
        if(Y[i]==lab):
            Y_lab.append(i)
            
    return len(list(set(X_lab).intersection(Y_lab)))/len(Y_lab),len(list(set(X_lab).intersection(Y_lab)))/len(X_lab)

def rSubset(arr, r):
    return list(combinations(arr,r))






if __name__ == '__main__':
    train,trainlabels,test,testlabels = loaddatasetMTS("ECG")
    with open('core_features_greedy', 'rb') as f:
        core_features = pickle.load(f)
    len(core_features)
    uniq_var = len(train[0])
    uniq_lab = set(trainlabels)
    classes = []#contants all shapelets classwise and variatevise shape -> classes x variate x shapelets 
    for i in range(len(uniq_lab)):
        temp = []
        for j in range(uniq_var):
            temp.append([])
        classes.append(temp)
            
    for i in core_features:
        classes[i[1]-1][i[5]].append(i)
        
    mtxs = []
    for cls in classes:
        tempdata = []
        for s in range(len(train)):
            temp = []
            for c in range(len(cls)):
                var = []
                for core in range(len(cls[c])):
                    if(min_euclidean(train[s][c],cls[c][core][0])<cls[c][core][2]):
                        var.append(1)
                    else:
                        var.append(0)
                temp.append(var)
            tempdata.append(temp)
        mtxs.append(tempdata)
        
    core_rules = []
    
    for u in range(len(uniq_lab)):    
        mtsVar = uniq_var
        varIndex=[int(j) for j in range(uniq_var)]
        mtsClassShapelets = []
        for k in range(uniq_var):
            mtsClassShapelets.append(len(classes[u][k]))
        variatesComninations=[]  # All combinations of variables 
        Rules=[]
        
        for i in range(1,mtsVar+1):
            rset=rSubset(varIndex,i)
            for r in rset:
                variatesComninations.append((r))
            
        for varSet in variatesComninations:
            lenSet=len(varSet)
            rules=[]
            flag=True
            for i in range(lenSet):
                if(len(rules)==0):
                    for j in range(mtsClassShapelets[varSet[i]]):
                        rules.append((varSet[i],j))
                else:
                    tempRules=[]
                    for rule in rules:
                        for j in range(mtsClassShapelets[varSet[i]]):
                            if flag == True:
                                tempRules.append((rule,(varSet[i],j)))
                            else :
                                rule1=list(rule)
                                rule1.append((varSet[i],j))
                                rule1=tuple(rule1)
                                tempRules.append(rule1)
                    rules=tempRules
                    flag=False
            Rules.extend(rules)
        
        Rules_class = []
        for i in Rules:
            if(isinstance(i[0], tuple)):
                temp = []
                for j in i:
                    temp.append(j)
                Rules_class.append(list(temp))
            else:
                temp = []
                temp.append(i)
                Rules_class.append(temp)
        
        rules_all = []
        for i in Rules_class:
            temp = []
            for j,k in i:
                temp.append(classes[u][j][k])
            rules_all.append(temp)
    
        for tr in rules_all:
            for trr in tr:
                print(trr[1],end=",")
            print()
        for dd in range(len(Rules_class)):
            print(dd,Rules_class[dd])
        
        
        
        all_ans = []
        quality = [[y for x in range(1)] for y in range(len(rules_all))]
        count=0
        for rule in Rules_class:
            temp = [int(0) for i in range(len(train))]
            flag=1
            for d in range(len(train)):
                for l,m in rule:
                    if(mtxs[u][d][l][m]==1):
                        flag=1
                    else:
                        flag=0
                        break
                if(flag==1):
                    temp[d]=u+1    
            all_ans.append(temp)
            
            pre,rec = prec(trainlabels, temp, u+1)
            
            if(pre<0.51 or rec<0.10):
                quality[count].insert(0,0)
            else:
                quality[count].insert(0,2*pre*rec/(pre+rec))
            count+=1
    
        quality.sort(reverse= True)
        for dd in range(len(quality)):
            print(dd, quality[dd])
        
        data_arr = []
        for data_lab in range(len(train)):
            if(trainlabels[data_lab]==u+1):
                data_arr.append(data_lab)
        
        print("data_arr is", data_arr)
        
        for q in quality:
            if(q[0]!=0):
                temp = []
                for h in range(len(train)):
                    if(all_ans[q[1]][h]==u+1 and h in data_arr):
                        temp.append(h)
                for h in temp:
                    if(h in data_arr):
                        data_arr.remove(h)
                print("remaining data",data_arr)    
                if(len(temp)!=0):
                    print("appending",len(temp))
                    core_rules.append([u+1,Rules_class[q[1]]])
                else:
                    print("empty")
                        
    early = 0
    test_pred = []
    earliness = 0
    for s in range(test.shape[0]):
        for L in range(len(test[s][0])+1):
            for c in core_rules:
                flag = 1
                for j,k in c[1]:
                    if(min_euclidean(test[s][j][0:L],classes[c[0]-1][j][k][0])<classes[c[0]-1][j][k][2]):
                        flag = 1
                    else:
                        flag = 0
                        break
                if(flag == 1):
                    lab = c[0]
                    break
                else:
                    lab = 0
            
            if(flag==1):
               # print(L,len(test[s][0]))
                early = L/len(test[s][0])
                break
        print(L,len(test[s][0]))
        earliness += (1-(early))
        test_pred.append(lab)
        
    count_p = 0
    count_a = 0
    for i in range(len(test)):
        if(test_pred==0):
            count_a += 1 
        if(test_pred[i]==testlabels[i]):
            count_p += 1
    print("accuracy on pred data is", count_p/(len(test)-count_a))
    print("earliness ", earliness/(len(test)-count_a))