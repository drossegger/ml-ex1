from sklearn.linear_model import SGDClassifier
from routines.preprocess import *
import numpy as np

def CalcSGD(traindata, trainlabel, testdata, testlabel, preprocess_method):
	traindata=preprocess_apply(traindata, preprocess_method)
	clf = SGDClassifier(loss="log", penalty="l2",shuffle=True)
	trainlabel=[repr(a) for a in trainlabel]
	clf.fit(traindata,trainlabel)
		
	testdata=preprocess_apply(testdata, preprocess_method)
	prediction=[]
	for testrecord in testdata :
		prediction.append( clf.predict(testrecord).tolist()[0])
		
	print 'Stochastic Gradient Descent :'
	print prediction
	print testlabel
	diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
	print diff
	print np.mean(diff)
