from sklearn import linear_model
from routines.datareader import readCSV
from routines.preprocess import preprocess_applymean
import numpy as np


def CalcBayesianRidge():
	data=readCSV('data/auto-mpg.data', range(1,7), 0)
	attributeList=preprocess_applymean(data[1], 2)
	clf = linear_model.BayesianRidge()
	label=[float(a) for a in data[0]]
	clf.fit(attributeList,label)
	
	
	testdata=readCSV('data/auto-mpg-predictors.data', range(1,7), 0)
	
	
	testinputs=preprocess_applymean(testdata[1], 2)
	prediction=[]
	for testinput in testinputs :
		#predict mpg values and print prediction
		#print clf.predict([testinput[1],testinput[2]])
		prediction.append( clf.predict(testinput))
	print 'Bayesian Ridge Regression :'
	print prediction
	print testdata[0]
	diff=[float(a)-float(b) for a,b in zip(prediction,testdata[0])]
	print diff
	print np.mean(diff)
