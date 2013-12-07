from sklearn import tree
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase
from base.constants import Constants

class DecisionTree(algorithmbase):		
	
	def ExtraParams(self, criterion):
		self.criterion = criterion
		return self
	
	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.missingvaluemethod, self.preprocessingmethods)
		
		
	def PrepareModel(self, savedmodel = None):
		
		if savedmodel != None:
			self.clf = savedmodel
		else:
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION: 
				self.clf=tree.DecisionTreeRegressor()
			elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				self.clf=tree.DecisionTreeClassifier(self.criterion)
			
			self.clf.fit(self.traindata ,self.trainlabel)
		
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.missingvaluemethod, self.preprocessingmethods)
			

	def Predict(self):
		prediction=[]
		for testrecord in self.testdata :
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION: 
				prediction.append( self.clf.predict(testrecord)[0])
			elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				prediction.append( float(self.clf.predict(testrecord)[0]))
			
			
			
		self.result = 	[self.testlabel, prediction]
		
	def GetModel(self):
		return self.clf