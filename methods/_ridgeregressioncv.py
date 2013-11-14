from sklearn import linear_model
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase

class RidgeRegressionCV(algorithmbase):		
	
	def ExtraParams(self, alphas):
		self.alphas = alphas
		return self
	
	
	
	def DoWork(self):
		self.traindata=preprocess_apply(self.traindata, self.preprocess_method)
		clf = linear_model.RidgeCV(alphas=self.alphas)
		clf.fit(self.traindata,self.trainlabel)
		
		
		testdata=preprocess_apply(self.testdata, self.preprocess_method)
		prediction=[]
		for testrecord in testdata :
			prediction.append( clf.predict(testrecord))
			
		self.result = [self.testlabel, prediction]