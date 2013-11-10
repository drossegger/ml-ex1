from sklearn import linear_model
from routines.datareader import readCSV
from routines.preprocess import preprocess_apply
import numpy as np


def CalcBayesianRidge(data_attributeList, data_label, test_attributeList, test_label, preprocess_method):
	
	data_attributeList=preprocess_apply(data_attributeList, preprocess_method)
	clf = linear_model.BayesianRidge()
	clf.fit(data_attributeList,data_label)
	
	
	testinputs=preprocess_apply(test_attributeList, preprocess_method)
	prediction=[]
	for testinput in testinputs :
		#predict mpg values and print prediction
		#print clf.predict([testinput[1],testinput[2]])
		prediction.append( clf.predict(testinput))
	print 'Bayesian Ridge Regression :'
	print prediction
	print test_attributeList
	diff=[float(a)-float(b) for a,b in zip(prediction,test_label)]
	print diff
	print np.mean(diff)
