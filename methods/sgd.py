from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase
from base.constants import Constants

class SGD(algorithmbase):
	
	def ExtraParams(self,loss='squared_loss',epsilon=10):
		self.loss=loss
		self.epsilon=epsilon
		return self
	
	
	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.missingvaluemethod, self.preprocessingmethods)
		self.trainlabel=[repr(a) for a in self.trainlabel]
		
	def PrepareModel(self, savedmodel = None):
				
		if savedmodel != None:
			self.clf = savedmodel
		else:
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION: 
				if self.loss=='squared_loss':
					self.clf=SGDRegressor(loss=self.loss, penalty="l2",shuffle=True)
				else:
					self.clf=SGDRegressor(loss=self.loss,epsilon=self.epsilon, penalty="l2",shuffle=True)
			elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				if self.loss=='squared_loss':
					self.clf=SGDClassifier(loss=self.loss, penalty="l2",shuffle=True)
				else:
					self.clf=SGDClassifier(loss=self.loss,epsilon=self.epsilon, penalty="l2",shuffle=True)
			self.clf.fit(self.traindata ,self.trainlabel)
		
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.missingvaluemethod, self.preprocessingmethods)
			

	def Predict(self):
		prediction=[]

		for testrecord in self.testdata :
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION: 
				prediction.append( self.clf.predict(testrecord))
			elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				prediction.append( float(self.clf.predict(testrecord)[0]))
			
		self.result = 	[self.testlabel, prediction]
		
	def GetModel(self):
		return self.clf
