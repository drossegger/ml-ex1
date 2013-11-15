from sklearn import linear_model
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase
import cPickle

class BayesianRidgeRegression(algorithmbase):
		
		
	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.preprocess_method)
		
		
	def PrepareModel(self, savedmodel = None):
		
		if savedmodel != None:
			self.clf = savedmodel
		else:
			self.clf = linear_model.BayesianRidge()
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