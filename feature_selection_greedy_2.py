'''loading all shapelets, quality,datasets '''
with open('features', 'rb') as f:
    features = pickle.load(f)
with open('output', 'rb') as f:
    shapelets = pickle.load(f)
with open('train', 'rb') as f:
    train = pickle.load(f)
with open('trainlabels', 'rb') as f:
    trainlabels = pickle.load(f)

	
''''''
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

with open('core_features', 'wb') as f:
    pickle.dump(core_feature,f)
