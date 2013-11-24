from sklearn import svm
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase
from base.constants import Constants

class SupportVectorMachine(algorithmbase):
	
	def ExtraParams(self,kernel='rbf',C=0.1):
		self.kernel=kernel
		self.C=C
		return self
	
	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.preprocess_method)
		self.trainlabel=[repr(a) for a in self.trainlabel]
		
	def PrepareModel(self, savedmodel = None):
				
		if savedmodel != None:
			self.clf = savedmodel
		else:
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION:
				self.clf=svm.SVR(C=self.C, kernel=self.kernel) 
			elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				self.clf=svm.SVC(C=self.C, kernel=self.kernel)
			self.clf.fit(self.traindata ,self.trainlabel)
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.preprocess_method)
			

	def Predict(self):
		prediction=[]
			
		if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION: 
			for testrecord in self.testdata :
				prediction.append( self.clf.predict(testrecord))
		elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
			for testrecord in self.testdata :
				prediction.append( self.clf.predict(testrecord)[0])
		
		self.result = 	[self.testlabel, prediction]
		
	
	def GetModel(self):
		return self.clf