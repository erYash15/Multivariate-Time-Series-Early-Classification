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
from sklearn.metrics import f1_score
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
    
    
with open('core_features', 'rb') as f:
    core_features = pickle.load(f)
    
    
uniq_var = len(train[0])
uniq_lab = set(trainlabels)
classes = []

for i in range(len(uniq_lab)):
    temp = []
    for j in range(uniq_var):
        temp.append([])
    classes.append(temp)
        
for i in core_features:
    classes[i[1]-1][i[5]].append(i)

#print(classes)

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
def rSubset(arr, r):
    return list(combinations(arr,r))
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
    
    #print("ss",Rules_class)
    rules_all = []
    for i in Rules_class:
        temp = []
        for j,k in i:
            print(u,j,k)
            temp.append(classes[u][j][k])
        rules_all.append(temp)
    
    print(rules_all)
    all_ans = []
    quality = [[y for x in range(1)] for y in range(len(rules_all))]
    count=0
    for rule in Rules_class:
        #print(rule[0][1],len(rule))
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
        print(temp)
        all_ans.append(temp)
        quality[count].insert(0,f1_score(trainlabels, temp, average='micro'))
        count+=1

    quality.sort(reverse= True)
    print(quality) 
    
    data_arr = [int(i) for i in range(len(train))]
    for q in quality:
        temp = []
        for h in range(len(train)):
            if(all_ans[q[1]][h]==u+1 and h in data_arr):
                temp.append(h)
        for h in temp:
            if(h in data_arr):
                data_arr.remove(h)
        if(len(temp)!=0):
            core_rules.append([u+1,Rules_class[q[1]]])
            
            
#testing
early = 0
test_pred = []
earliness = 0
for s in range(test.shape[0]):
    for L in range(len(test[s][0])):
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
            print(L,len(test[s][0]))
            early = L/len(test[s][0])
            break
    earliness += (1-(early))
    test_pred.append(lab)
print(test_pred,earliness)