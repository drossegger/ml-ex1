from sklearn import linear_model
from routines.preprocess import preprocess_apply
import numpy as np


def CalcBayesianRidgeRegression(traindata, trainlabel, testdata, testlabel, preprocess_method):
	
	traindata=preprocess_apply(traindata, preprocess_method)
	clf = linear_model.BayesianRidge()
	clf.fit(traindata,trainlabel)
	
	
	testdata=preprocess_apply(testdata, preprocess_method)
	prediction=[]
	for testrecord in testdata :
		prediction.append( clf.predict(testrecord))
		
	return [testlabel, prediction]
