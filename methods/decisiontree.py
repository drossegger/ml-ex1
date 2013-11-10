from sklearn import tree
from routines.preprocess import preprocess_apply
import numpy as np


def CalcDecisionTree(traindata, trainlabel, testdata, testlabel, preprocess_method):
	
	traindata=preprocess_apply(traindata, preprocess_method)
	clf=tree.DecisionTreeRegressor()
	clf.fit(traindata,trainlabel)
	
	
	testdata=preprocess_apply(testdata, preprocess_method)
	prediction=[]
	for testrecord in testdata :
		prediction.append( clf.predict(testrecord))
		
	diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
	diffmean=np.mean(diff)
	
	print 'Decision Tree:'
	print testlabel
	print prediction
	print diff
	print diffmean
	return diffmean
