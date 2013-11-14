from sklearn import svm
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase

class SupportVectorMachine(algorithmbase):
	
	def DoWork(self):
		
		self.traindata=preprocess_apply(self.traindata, self.preprocess_method)
		clf = svm.SVR()
		clf.fit(self.traindata,self.trainlabel)
		
		
		testdata=preprocess_apply(self.testdata, self.preprocess_method)
		prediction=[]
		for testrecord in testdata :
			prediction.append( clf.predict(testrecord))
			
		self.result = [self.testlabel, prediction]
	