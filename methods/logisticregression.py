from sklearn import linear_model
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase
from base.constants import Constants

class LogisticRegression(algorithmbase):		
	
	def ExtraParams(self, C):
		self.C = C
		return self
	
		
	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.preprocess_method)
		
		
	def PrepareModel(self, savedmodel = None):
		
		if savedmodel != None:
			self.clf = savedmodel
		else:
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				self.clf=linear_model.LogisticRegression(C=self.C)		
				self.clf.fit(self.traindata ,self.trainlabel)
		
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.preprocess_method)
			

	def Predict(self):
		prediction=[]
		for testrecord in self.testdata :
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				prediction.append( float(self.clf.predict(testrecord)[0]))
			
		self.result = [self.testlabel, prediction]
		
	def GetModel(self):
		return self.clf
