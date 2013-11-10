from sklearn import neighbors
from routines.preprocess import preprocess_apply
import numpy as np


def CalcNearestNeighborsRegression(traindata, trainlabel, testdata, testlabel, preprocess_method, n_neighbors, weight):
	
	traindata=preprocess_apply(traindata, preprocess_method)
	knn=neighbors.KNeighborsRegressor(n_neighbors, weights=weight)
	knn.fit(traindata,trainlabel)
	
	testdata=preprocess_apply(testdata, preprocess_method)

	prediction=[]
	for testrecord in testdata :
		prediction.append( knn.predict(testrecord))
		
	return [testlabel, prediction]
