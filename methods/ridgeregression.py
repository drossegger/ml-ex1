from sklearn import linear_model
from routines.preprocess import preprocess_apply
import numpy as np


def CalcRidgeRegression(traindata, trainlabel, testdata, testlabel, preprocess_method, alpha):
	
	traindata=preprocess_apply(traindata, preprocess_method)
	clf = linear_model.Ridge(alpha = alpha)
	clf.fit(traindata,trainlabel)
	
	
	testdata=preprocess_apply(testdata, preprocess_method)
	prediction=[]
	for testrecord in testdata :
		prediction.append( clf.predict(testrecord))
		
	diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
	diffmean=np.mean(diff)
	print 'Ridge Regression :'
	print testlabel
	print prediction
	print diff
	print diffmean
	return diffmean
