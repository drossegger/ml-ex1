from sklearn.linear_model import SGDClassifier
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase

class SGD(algorithmbase):
	
	def ExtraParams(self,loss='hinge'):
		self.loss=loss
		return self
	
	
	def DoWork(self):
		'''
		Deprecated
		'''
		self.traindata=preprocess_apply(self.traindata, self.preprocess_method)
		clf = SGDClassifier(loss=self.loss, penalty="l2",shuffle=True)
		self.trainlabel=[repr(a) for a in self.trainlabel]
		clf.fit(self.traindata,self.trainlabel)
			
		testdata=preprocess_apply(self.testdata, self.preprocess_method)
		prediction=[]
		for testrecord in testdata :
			prediction.append( clf.predict(testrecord))
			
		self.result = [self.testlabel, prediction]


	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.preprocess_method)
		self.trainlabel=[repr(a) for a in self.trainlabel]
		
	def PrepareModel(self, savedmodel = None):
				
		if savedmodel != None:
			self.clf = savedmodel
		else:
			self.clf=SGDClassifier(loss=self.loss, penalty="l2",shuffle=True)
			self.clf.fit(self.traindata ,self.trainlabel)
		
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.preprocess_method)
			

	def Predict(self):
		prediction=[]
		for testrecord in self.testdata :
			prediction.append( self.clf.predict(testrecord))
			
		self.result = 	[self.testlabel, prediction]
		
	def GetModel(self):
		return self.clf