'''without warping window'''
def dtw_wow(X,Y):
    w, h = (len(X)), (len(Y))
    dtw = [[0 for x in range(w)] for y in range(h)] 
    for i in range(w):
        for j in range(h):
            dist = abs(float(Y[j])-float(X[i]))
        if(i-1<0):
            if(j-1<0):
                minimum = 0
            else:
                minimum = dtw[i][j-1]
        elif(j-1<0):
            minimum = dtw[i-1][j]
        else:
            minimum = min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        dtw[i][j] = dist + minimum
    return dtw[w-1][h-1]
	
	
'''with warping window'''#code may require modifications
def dtw_ww(X,Y,window):
    window = max(window, len(X))
    w, h = (len(X)), (len(Y))
    dtw = [[0 for x in range(w)] for y in range(h)] 
    for i in range(w):
        for j in range(max(1,i-window),min(len(X),i+window)):
            dist = abs(float(Y[j])-float(X[i]))
        if(i-1<0):
            if(j-1<0):
                minimum = 0
            else:
                minimum = dtw[i][j-1]
        elif(j-1<0):
            minimum = dtw[i-1][j]
        else:
            minimum = min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        dtw[i][j] = dist + minimum
    return dtw[w-1][h-1]