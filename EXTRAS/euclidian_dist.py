import csv



with open('synthetic_control_TRAIN.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    readCSVlistTRAIN = list(readCSV)
print(len(readCSVlistTRAIN[0]),len(readCSVlistTRAIN))
with open('synthetic_control_TEST.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    readCSVlistTEST = list(readCSV)
print(len(readCSVlistTEST[0]),len(readCSVlistTEST))
readCSVlistTRAINlabels = []
for i in range(len(readCSVlistTRAIN)):
    readCSVlistTRAINlabels.append(readCSVlistTRAIN[i][0])
readCSVlistTRAINfeatures = []
for i in range(len(readCSVlistTRAIN)):
    readCSVlistTRAINfeatures.append(readCSVlistTRAIN[i][1:])
readCSVlistTESTlabels = []
for i in range(len(readCSVlistTEST)):
    readCSVlistTESTlabels.append(readCSVlistTRAIN[i][0])

	
	
correct = 0
for i in range(len(readCSVlistTEST)):
    best = float("inf")
    pred_class = 0
    dist = 0
    for j in range(len(readCSVlistTRAIN)):
        sum = 0
        for k in range(len(readCSVlistTRAINfeatures[1])):
            sum = sum + ( float(readCSVlistTRAINfeatures[j][k]) - float(readCSVlistTESTfeatures[i][k]))**2
        dist = sum**(0.5)
        if(dist < best):
            best = dist
            pred_class = readCSVlistTRAINlabels[j]
      #  print(pred_class, int(readCSVlistTESTlabels[i]))   
    if(int(pred_class) == int(readCSVlistTESTlabels[i])):
        correct = correct + 1

		
print(correct)
error_rate = (len(readCSVlistTEST) - correct)/len(readCSVlistTEST)
print(error_rate)
