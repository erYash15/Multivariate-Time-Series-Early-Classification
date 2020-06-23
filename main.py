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
import statistics
import random
from itertools import combinations
from statistics import mean
from math import ceil

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

'''feature extraction'''

def euclidian(X,Y):
    #Input: array X ( M x 1 ) and array Y ( M x 1)
    #Output: euclidian distance between 
    return np.linalg.norm(X-Y)
	
def min_euclidean(X,Y):
    #Input: array X and array Y 
    #X=Mx1, Y=Nx1
    #Output: minimum euclidian distance of Y from X
    min = float("inf")
    for i in range(len(X)-len(Y)+1):
        dist=euclidian(np.array(X[i:i+len(Y)]),np.array(Y))
        if(dist<min):
            min = dist
    return(min)
	
def Learnthreshold(candidate,data,labels,ss,tt):
    ###Input###
    #tt is variate of shapelets
    #ss is data sample to which shapelet belong
    #candidate: Px1, data: MxVxT, labels: Mx1, s: integer value, t:integer value
    #data: training data M x V x L
    #Output: List with threshold, precision, recall and quality   
    dist_mtx = [] 
    thre_cand = []
    sorted_labels = [] # for finding precision and recall after sorted dist_matrix 
    #so that labels don't lose
    total_c = np.count_nonzero(labels == labels[ss]) #count total sample points in train data having same label as shapelet
    for s in range(data.shape[0]):
        dist_mtx.append([min_euclidean(data[s][tt],candidate),labels[s]])
    dist_mtx.sort()
    for i in dist_mtx:
        sorted_labels.append(i[1])
    for i in range(len(dist_mtx)-1):
        threshold = (dist_mtx[i][0]+dist_mtx[i+1][0])/2
        precision = (sorted_labels[0:i+1].count(labels[ss]))/len(dist_mtx[0:i+1])#precision by using formaula in paper
        recall = (sorted_labels[0:i+1].count(labels[ss]))/total_c # recall by using formula in paper
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
    ##Input:## 
    #data: training data dimension: M x V x L
    #labels: training labels array M x 1
    #Lmin: minimum length (int)
    #Pre_min: minimum precision of any shapelet(int)
    #Rec_min: Minimum Recall for any shapelet(int)
    ##Output##:
    #list of list: 
    # output : |No_of_allFea|x 8
    # |No_of_allFea|[0]: Shapelet
    # |No_of_allFea|[1]: Class Label
    # |No_of_allFea|[2]: Threshhold
    # |No_of_allFea|[3]: Precision
    # |No_of_allFea|[4]: Recall
    # |No_of_allFea|[5]: Variate
    # |No_of_allFea|[6]: Lenght
    # |No_of_allFea|[7]: sample no. in t to which shapelet belongs
    for s in range(data.shape[0]): # loop for data points
        print(s,end =',')
        for t in range(data[s].shape[0]):#Loop for variates in data points
            Lmax = len(data[s][t])//3
            #Lmax = 6
            for L in range(Lmin,Lmax+1): #Loop for length for shapelets 
                # example min=4 and max=7, range(min,max+1)={4,5,6,7}                
                for i in range(0,len(data[s][t])-L+1):# loop for starting index of timeseries 
                    candidate = data[s][t][i:i+L]
                    threshold_info = Learnthreshold(candidate,data,labels,s,t)
                    if(threshold_info[1]>=Pre_min and threshold_info[2]>=Rec_min):
                        output.append([candidate,labels[s],threshold_info[0],threshold_info[1],threshold_info[2],t,L,s])

def min_euclidean_2(X,Y):
    #Input: X=Mx1, Y=Nx1
    #Output: minimum euclidian distance of smaller array to larger array 
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

def fss_feature_extraction(train,trainlabels,Lmin=5,Pre_min=0.51,Rec_min=0.10):
    ##Input:## 
    #data: training data dimension: M x V x L
    #labels: training labels array M x 1
    #Lmin: minimum length (int)
    #Pre_min: minimum precision of any shapelet(int)
    #Rec_min: Minimum Recall for any shapelet(int)
    ##Output##:
    #list of list: 
    # output : |No_of_allFea|x 8
    # |No_of_allFea|[0]: Shapelet
    # |No_of_allFea|[1]: Class Label
    # |No_of_allFea|[2]: Threshhold
    # |No_of_allFea|[3]: Precision
    # |No_of_allFea|[4]: Recall
    # |No_of_allFea|[5]: Variate
    # |No_of_allFea|[6]: Lenght
    # |No_of_allFea|[7]: sample no. in t to which shapelet belongs
    uniq_lab = set(trainlabels)
    uniq_var = len(train[0])
    data = [ [[] for j in range(uniq_var)] for i in range(len(uniq_lab))]
    print(data)
    #print(len(trainlabels))
    for s in range(train.shape[0]):
        for t in range(train[s].shape[0]):
            data[trainlabels[s]-1][t].append(train[s][t])  
    active = [ [0 for i in range(train.shape[0])] for j in range(train[0].shape[0])]
    #print(active)
    #active: dimension V x M where V is variate and M is data point 
    L=1
    bkrcount = 0
    bkr = 0
    #print(len(data))
    for s in range(len(data)):
        print(s)
        for t in range(len(data[s])):
            print(s,t)
            sum_arr = []
            for ser in range(len(data[s][t])):
                sum_arr.append(sum(data[s][t][ser])/len(data[s][t][ser]))
            mean_ser = statistics.mean(sum_arr)
            diff_sum_mean = []
            for ser in range(len(sum_arr)):
                diff_sum_mean.append(sum_arr[ser]-mean_ser)
            ts_idx = diff_sum_mean.index(min(diff_sum_mean))
            ecld_dists = []
            for ser in range(len(data[s][t])):
                dist = min_euclidean_2(data[s][t][ts_idx],data[s][t][ser])
                ecld_dists.append([dist,ser])
            print(ecld_dists)
            ecld_dists.sort()
            adv_disp = []
            
            for ser in range(len(ecld_dists)-1):
                adv_disp.append(ecld_dists[ser][0]-ecld_dists[ser+1][0])
        
            std_dev = statistics.stdev(adv_disp)
            half_std_dev = std_dev/2
            
            seprator = [-1]
            
            for ser in range(len(ecld_dists)-1):
                if(ecld_dists[ser+1][0]-ecld_dists[ser][0]>half_std_dev):
                    seprator.append(ser)
            print(seprator)
            #here we increase the index ("bkr") to make "active" list element 1 only when next class data comes. 
            if(s!=0 and t==0):
                bkr += len(data[bkrcount][0])
                bkrcount += 1 
                print("brks",bkr,bkrcount)
            #loop make the useful data 1 rest remain 0 for index of perticular class.
            for sep in seprator:
                if(s==0):
                    active[t][ecld_dists[sep+1][1]]=1
                else:
                    active[t][ecld_dists[sep+1][1]+bkr]=1
    
    #just the counting function that counts the active timeseries in the MTS data
    count = 0
    for kk in active:
        for nn in range(len(kk)):
            if(kk[nn]==1):
                print(nn)
                count+=1
    print("count", count)
    
    for s in range(train.shape[0]):# loop for data points
        print(s,end =',')
        for t in range(train[s].shape[0]):#Loop for variates in data points
            if(active[t][s]==1):
                Lmax = len(train[s][t])//3
                #Lmax = 20
                for L in range(Lmin,Lmax+1):  #Loop for length for shapelets 
                    # example min=4 and max=7, range(min,max+1)={4,5,6,7}                
                    for i in range(0,len(train[s][t])-L+1): # loop for starting index of timeseries
                        candidate = train[s][t][i:i+L]
                        threshold_info = Learnthreshold(candidate,train,trainlabels,s,t)
                        if(threshold_info[1]>=Pre_min and threshold_info[2]>=Rec_min):
                            output.append([candidate,trainlabels[s],threshold_info[0],threshold_info[1],threshold_info[2],t,L,s])
                    

'''feature selction'''

def MIL(shapelet_local,X):
    ##Input: all shapelets
    #shapelet_local[0] = shapelet
    #shapelet_local[2] is threshold
    ##Output: if match then length else "NAN"
    for i in range(len(X)-len(shapelet_local[0])+1):
        dist=np.linalg.norm(np.array(X[i:i+len(shapelet_local[0])]-np.array(shapelet_local[0])))
        if(dist<shapelet_local[2]):
            return i+len(shapelet_local[0])
    return "NAN"
    #returns the MIL distance if fouund else total length
	
def earli_cal(shapelet_local,data):
    ##Input: all shapelets
    #data: training data
    #shapelet_local[5] is variate
    ##Output: earlinesss
    earliness = 0
    count = 0
    for s in range(data.shape[0]):
        mil_length = MIL(shapelet_local,data[s][int(shapelet_local[5])])
        if(mil_length != "NAN"):
            early = mil_length/len(data[s][int(shapelet_local[5])])
            earliness += (1 - early)
            count += 1
    if(count==0):
        return earliness
    earliness = earliness/count
    return earliness


def feature_selection(shapelets_local,data,labels,w0=1,w1=1,w2=1):
    #shapelets_local is list of list of shapelets with information 
    #|No_of_allFea|x 8
    # |No_of_allFea|[0]: Shapelet
    # |No_of_allFea|[1]: Class Label
    # |No_of_allFea|[2]: Threshhold
    # |No_of_allFea|[3]: Precision
    # |No_of_allFea|[4]: Recall
    # |No_of_allFea|[5]: Variate
    # |No_of_allFea|[6]: Lenght
    # |No_of_allFea|[7]: sample no. in t to which shapelet belongs
    GEFM_lst = [[y for x in range(2)] for y in range(len(shapelets_local))]
    #feature [|No_of_allFea| x 2]
    counter = 0
    for shapelet_local in shapelets_local:#for each shapelet in all shaplets
        #print(counter)
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
    GEFM_lst.sort(reverse= True)#sort based upon the first index
    #print(GEFM_lst)
    return GEFM_lst

def train_shapelets_pred(shapelet_local,Y):
    #Input: shapelet_local: shapelet of (8 x * )
    #shapelet_local[0] is shapelet
    #shapelet_local[2] is threshold
    #output: return the predicted label on data Y by shapelet shapelet_local
    for i in range((len(Y)-len(shapelet_local[0]))+1):
        dist=euclidian(np.array(Y[i:i+len(shapelet_local[0])]),np.array(shapelet_local[0]))
        if(dist<=shapelet_local[2]):
            return shapelet_local[1]
    return "NAN"

def imp_features(shapelets_local,data,labels,features):
    #shapelets_local is list of list of shapelets with information 
    #|No_of_allFea|x 8
    # |No_of_allFea|[0]: Shapelet
    # |No_of_allFea|[1]: Class Label
    # |No_of_allFea|[2]: Threshhold
    # |No_of_allFea|[3]: Precision
    # |No_of_allFea|[4]: Recall
    # |No_of_allFea|[5]: Variate
    # |No_of_allFea|[6]: Lenght
    # |No_of_allFea|[7]: sample no. in t to which shapelet belongs
    #features - index of all the shaplets in sorted order of quality
    imp_shapelets = [] #collects the important shapelets
    uniq_var = len(data[0])
    all_data_variatewise = [] #virtual copy of data variatewise
    for _ in range(uniq_var):
        all_data_variatewise.append([i for i in range(len(data))]) 
    # ex -> all_data_variatewise = [[1,2,.....100][1,2.....100]]
    all_shapelets_variatewise = [] # only the index of shapelets from sorted features list .. variatewise
    for _ in range(uniq_var):
        all_shapelets_variatewise.append([])

    for g in features:
        all_shapelets_variatewise[shapelets_local[g[1]][5]].append(g[1])

    for variate in range(len(all_shapelets_variatewise)):
        for shapelets in all_shapelets_variatewise[variate]:
            #print(shapelets)
            prev_len = len(all_data_variatewise[variate])
            temp = [] # collectes the index of reduced data
            for i in all_data_variatewise[variate]:
                #print(shapelets,i,variate)
                pred = train_shapelets_pred(shapelets_local[shapelets],data[i][variate]) 
                if(pred == labels[i]):
                    temp.append(i) # append all index with can be removed
            for j in temp:
               all_data_variatewise[variate].remove(j) # remove the data from virtual copy that was present in temp
            curr_len = len(all_data_variatewise[variate])
            if(prev_len>curr_len): # select only those shapelets which actually reduce the data
                imp_shapelets.append(shapelets_local[shapelets])
                #print(all_data_variatewise)
    return imp_shapelets

def SI_clust(data,no_cluster,num_iters):
    ##Input: data -  [m x l]
    #no_cluster: number of cluster
    #num_iteration: number of iteration
    ##output: pred_labels predicted labels - (M x 1)
    #mean(SI) - mean of SI indexes
    #SI - SI index of all the data points - (M x 1)
    data_idxs = [[int(i), random.randint(0, no_cluster-1)] for i in range(len(data))]
    pred_labels = [ g[1] for g in data_idxs ]
    s_label = 0
    for _ in range(num_iters):
        SI = []
        closest_cluster = []
        for i in range(len(data)):
            dist = []
            for s in range(len(data)):#s iterate for all data
                dist.append(min_euclidean_2(data[i],data[s]))##distance is calculated for each point so that
                # does not give memory error
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
    

def SI_clustering(shapelets,train,trainlabels):
    ##Input: shapelets :- all shapelets dimension - |No_of_allFea|x 8
    # |No_of_allFea|[0]: Shapelet
    # |No_of_allFea|[1]: Class Label
    # |No_of_allFea|[2]: Threshhold
    # |No_of_allFea|[3]: Precision
    # |No_of_allFea|[4]: Recall
    # |No_of_allFea|[5]: Variate
    # |No_of_allFea|[6]: Lenght
    # |No_of_allFea|[7]: sample no. in t to which shapelet belongs
    #train: training data:M x V x L
    #trainlabels: M x 1
    ##Output: core features chossen from all the shapelets : |No_of_corFea|x 8
    
    uniq_lab = set(trainlabels)
    uniq_var = len(train[0])
    data = [ [[] for j in range(uniq_var)] for i in range(len(uniq_lab))]
    #print(data)
    for i in shapelets:
        data[i[1]-1][i[5]].append(i)
    core_features = []
    
    for cc in range(0,len(data)):
        for vv in range(0,len(data)):
            new_labels = []
            new_si_score = 0
            new_si_sample = []
            print(cc,vv)
            temp = [d[0] for d in data[cc][vv]]
            
            for k in range(3,4):
                #print("k",k)
                #print(temp,k)
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
    return core_features

'''classifiers'''
def prec(X,Y,lab):
    #X is list of true labels dimension (M x 1)
    #Y is list of predicted labels (N x 1)
    Y_lab = []
    X_lab = []
    for i in range(len(X)):
        if(X[i]==lab):
            X_lab.append(i)
        if(Y[i]==lab):
            Y_lab.append(i)
    try:
        pr = len(list(set(X_lab).intersection(Y_lab)))/len(Y_lab)
    except:
        pr = 0
    try:
        re = len(list(set(X_lab).intersection(Y_lab)))/len(X_lab)
    except:
        re = 0
    return pr,re

def rSubset(arr, r):
    return list(combinations(arr,r))

def rule_based(train,trainlabels,core_features):
    ##Input:#train: training data:M x V x L
    #trainlabels: M x 1
    # core_features : |No_of_CoreFea|x 8
    # |No_of_CoreFea|[0]: Shapelet
    # |No_of_CoreFea|[1]: Class Label
    # |No_of_CoreFea|[2]: Threshhold
    # |No_of_CoreFea|[3]: Precision
    # |No_of_CoreFea|[4]: Recall
    # |No_of_CoreFea|[5]: Variate
    # |No_of_CoreFea|[6]: Lenght
    # |No_of_CoreFea|[7]: sample no. in t to which shapelet belongs
    ##Output: 
    #accuracy on pred data
    #earliness on the pred datais
    #coverage
    #accuracy on complete data is
    #earliness on the complete data is
    uniq_var = len(train[0])
    uniq_lab = set(trainlabels)
    classes = []#contants all shapelets classwise and variatevise 
    #shape -> classes x variate x core_shapelets 
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
        
    cover = 0
    count_p = 0
    count_a = 0
    for i in range(len(test)):
        if(test_pred[i]==0):
            count_a += 1 
        else:
            cover += 1
        if(test_pred[i]==testlabels[i]):
            count_p += 1
    acc_pred = count_p/(len(test)-count_a)
    earl_pred = earliness/(len(test)-count_a)
    coverage = cover/(len(test))
    count_p = 0
    for i in range(len(test)):
        if(test_pred[i]==0):
            test_pred[i] = int(random.randint(1,len(set(trainlabels))))
        if(test_pred[i]==testlabels[i]):
            count_p += 1
    acc_full = count_p/(len(test))
    earl_full = (earliness+count_a)/(len(test))
    
    
    print("accuracy on pred data is", acc_pred)
    print("earliness on the pred datais", earl_pred)
    print("coverage", coverage)
    print("accuracy on complete data is", acc_full)
    print("earliness on the complete data is", earl_full)
    return  acc_pred,earl_pred,coverage,acc_full,earl_full

def max_feq(X):
    #Input: X is array of M * 1
    #Output: element with max freq or 0 if no unique element with max frequency
    try:
        res = statistics.mode(X)
    except:
        res = 0
    return res

def qbc(train,trainlabels,core_features):
    # train : M x V x T
    # trainlabels : M x 1
    # core_features : |No_of_CoreFea|x 8
    # |No_of_CoreFea|[0]: Shapelet
    # |No_of_CoreFea|[1]: Class Label
    # |No_of_CoreFea|[2]: Threshhold
    # |No_of_CoreFea|[3]: Precision
    # |No_of_CoreFea|[4]: Recall
    # |No_of_CoreFea|[5]: Variate
    # |No_of_CoreFea|[6]: Lenght
    # |No_of_CoreFea|[7]: sample no. in t to which shapelet belongs
    ##Output: 
    #accuracy on pred data
    #earliness on the pred datais
    #coverage
    #accuracy on complete data is
    #earliness on the complete data is
    pred_labels_flated = [] #dimension: M x V
    pred_labels_unflated = [] #dimension: M x V x 1
    earliness = 0
    early = [] #dimension: M x V x 2 [[[e,T]] e is early length while T is complete length
    uniq_var = len(train[0])
    #uniq_lab = set(trainlabels)
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
                    if(core_features[c][5]==t and len(core_features[c][0])<= L):
                       # print(min_euclidean(temp,core_features[c][0]) , core_features[c][2])
                        if(min_euclidean(temp,core_features[c][0]) < core_features[c][2]):
                            if(early[s][t][0]==early[s][t][1]):
                                early[s][t][0] = L
                                pred[t].append(core_features[c][1])
        flat_pred = [item for sublist in pred for item in sublist]
        pred_labels_flated.append(flat_pred)
        pred_labels_unflated.append(pred)
    #print(pred_labels_flated)
    
    
    '''predicting labels'''
    pred_labels_test_covered = []
    pred_labels_test_uncovered = []

    uncovered = 0
    for p in pred_labels_flated:
        print(p)
        if(max_feq(p)==0 or p.count(max_feq(p))<((uniq_var//2)+1)):
            try:
                pred_labels_test_uncovered.append(int(random.choice(p)))
                
            except:
                pred_labels_test_uncovered.append(int(random.randint(1,len(set(trainlabels)))))
            pred_labels_test_covered.append(0)
            uncovered += 1
        else:
            pred_labels_test_uncovered.append(max_feq(p))
            pred_labels_test_covered.append(max_feq(p))
    '''calculating earliness'''
    earliness = 0
    covered = 0
    for i in range(len(early)):
        if(pred_labels_flated[i].count(max_feq(pred_labels_flated[i])) >= (uniq_var//2)+1):
            Larr = []
            L = 0
            full_L = 0
            for k in range(uniq_var):
                if(pred_labels_unflated[i][k] == max_feq(pred_labels_flated[i])):
                    Larr.append(early[i][k][0])
                    full_L=max(full_L,early[i][k][1])
            Larr.sort()
            L = Larr[(uniq_var//2)]
            earliness += 1 - L/full_L
            covered +=1
        else:    
            L = 0
            full_L = 0
            for k in range(uniq_var):
                L = max(L,early[i][k][0])
                full_L=max(full_L,early[i][k][1])
            earliness += 0

    
    acc_pred = 0
    earl_pred = 0
    coverage = 0
    acc_full = 0
    earl_full = 0        
    '''finding accuracy with covered data(pred)'''		 
    count_a = 0
    count_p = 0
    count_allp = 0
    for i in range(len(test)):
        if(pred_labels_test_covered[i]==0):
            count_a += 1
        if(pred_labels_test_covered[i]==testlabels[i]):
            count_p += 1
    print(count_a)
    acc_pred = count_p/(len(test)-count_a)
    coverage = (len(test)-uncovered)/len(test)
    '''finding earl_pred'''
    
    print("d",earliness,covered,count_p)
    earl_pred = earliness/covered
    '''finding accuracy with uncovered data also(full)'''
    count_pf = 0
    for i in range(len(test)):
        if(pred_labels_test_uncovered[i]==testlabels[i]):
            count_pf += 1
    acc_full = count_pf/len(test)
    earl_full = earliness/len(test)
    
    
    print("accuracy on pred data is", acc_pred)
    print("earliness on the pred datais", earl_pred)
    print("coverage", coverage)
    print("accuracy on complete data is", acc_full)
    print("earliness on the complete datais", earl_full)
    return  acc_pred,earl_pred,coverage,acc_full,earl_full


if __name__ == "__main__":
    Dataset = "ECG"
    
    Fea_Ext = False
    Fea_Sel = False
    Clf     = True
    
    Method_1 = "fss_feature_extraction" # Options: "feature_extraction", "fss_feature_extraction"
    Method_2 = "si_clustering" # Options: "greedy", "si_clustering"
    Method_3 = "qbc" # Options: "qbc","rule_based"
    
    DATA_DIR = './'+str(Dataset)
    print("Dataset Name", DATA_DIR)
    path = os.path.join(DATA_DIR, Dataset)
    print("Working ",path)
    #loading data
    train,trainlabels,test,testlabels = loaddatasetMTS(Dataset)

    output = []
    
    '''2 Method of feature extraction'''
    '''Mehod 1'''
    if(Fea_Ext):
        if(Method_1 == "feature_extraction"):
            feature_extraction(train,trainlabels)
        else:
            fss_feature_extraction(train,trainlabels)
        file_name = "output_" + str(Method_1) + "_" + str(Dataset)    
        '''Saving shapelet file as output'''
        with open(file_name, 'wb') as f:
            pickle.dump(output, f)
    
    
    '''Method 2'''    
    '''reading all shapelets from data'''
    if(Fea_Sel):
        file_name = "output_" + str(Method_1) + "_" + str(Dataset)
        with open(file_name, 'rb') as f:
            shapelets = pickle.load(f)
        
        '''2 Method of feature selection'''
        if(Method_2 == "greedy"):
            features = feature_selection(shapelets,train,trainlabels)
            core_features = imp_features(shapelets,train,trainlabels,features)
        else:
            core_features = SI_clustering(shapelets,train,trainlabels)
            
        file_name = "core_features" + "_" +str(Method_1) + "_" + str(Method_2) + "_" + str(Dataset)
        with open(file_name, 'wb') as f:
            pickle.dump(core_features,f)
     
    
    '''Method 3'''
    '''readinf core_features'''
    if(Clf):
        file_name = "core_features" + "_" +str(Method_1) + "_" + str(Method_2) + "_" + str(Dataset)
        with open(file_name, 'rb') as f:
            core_features = pickle.load(f)
    
        if(Method_3 == "rule_based"):
            acc_pred,earl_pred,coverage,acc_full,earl_full = rule_based(train,trainlabels,core_features)
        else:
            acc_pred,earl_pred,coverage,acc_full,earl_full = qbc(train,trainlabels,core_features)
        file_name ="result" + "_" +str(Method_1) + "_" + str(Method_2) + "_" + str(Method_3) + "_" +str(Dataset)
        file = open(file_name,'w') 
        file.write("accuracy on predicted: "+str(acc_pred))
        file.write("\nearliness on predicted: "+str(earl_pred))
        file.write("\ncoverage: "+str(coverage))
        file.write("\naccuracy on complete: "+str(acc_full))
        file.write("\nearliness on predicted: "+str(earl_full))
        file.close() 