from sklearn.linear_model import SGDClassifier
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase

class SGD(algorithmbase):
	
	def ExtraParams(self,loss='hinge'):
		self.loss=loss
		return self
	def DoWork(self):
		self.traindata=preprocess_apply(self.traindata, self.preprocess_method)
		clf = SGDClassifier(loss=self.loss, penalty="l2",shuffle=True)
		self.trainlabel=[repr(a) for a in self.trainlabel]
		clf.fit(self.traindata,self.trainlabel)
			
		testdata=preprocess_apply(self.testdata, self.preprocess_method)
		prediction=[]
		for testrecord in testdata :
			prediction.append( clf.predict(testrecord).tolist()[0])
			
		self.result = [self.testlabel, prediction]
