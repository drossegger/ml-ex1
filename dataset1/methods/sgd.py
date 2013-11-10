from sklearn.linear_model import SGDClassifier
from routines.datareader import readCSV
from routines.preprocess import preprocess_applymean
import numpy as np

def CalcSGD():
	
	data=readCSV('data/auto-mpg.data', range(1,7), 0)
	#build model based on features 1 and 2 (horsepower and cylinders)
	attributes=preprocess_applymean(data[1], 2)
	#x=[[a[1],a[2]] for a in x]
	clf = SGDClassifier(loss="log", penalty="l2",shuffle=True)
	clf.fit(attributes,data[0])
		
	testdata=readCSV('data/auto-mpg-predictors.data', range(1,7), 0)
		
	testinputs=preprocess_applymean(testdata[1], 2)
	prediction=[]
	for testinput in testinputs :
		#predict mpg values and print prediction
		#print clf.predict([testinput[1],testinput[2]])
		prediction.append( clf.predict(testinput).tolist()[0])
	print 'Stochastic Gradient Descent :'
	print prediction
	print testdata[0]
	diff=[float(a)-float(b) for a,b in zip(prediction,testdata[0])]
	print diff
	print np.mean(diff)
