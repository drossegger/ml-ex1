from sklearn.linear_model import SGDClassifier
import routines
import numpy as np

def DoWork():
	
	data=routines.readCSV('data/auto-mpg.data')
	#build model based on features 1 and 2 (horsepower and cylinders)
	x=routines.preprocess(data[1])
	#x=[[a[1],a[2]] for a in x]
	clf = SGDClassifier(loss="log", penalty="l2",shuffle=True)
	clf.fit(x,data[0])
	
	
	testdata=routines.readCSV('data/auto-mpg-predictors.data')
	
	
	testinputs=routines.preprocess(testdata[1])
	prediction=[]
	for testinput in testinputs :
		#predict mpg values and print prediction
		#print clf.predict([testinput[1],testinput[2]])
		prediction.append( clf.predict(testinput).tolist()[0])
	print prediction
	print testdata[0]
	diff=[float(a)-float(b) for a,b in zip(prediction,testdata[0])]
	print diff
	print np.mean(diff)
