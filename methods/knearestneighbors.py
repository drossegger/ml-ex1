from sklearn import neighbors
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase
from base.constants import Constants

class KNearestNeighbors(algorithmbase):		
	
	def ExtraParams(self, n_neighbors, weight):
		self.n_neighbors = n_neighbors
		self.weight = weight
		return self
	
	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.preprocess_method)
		
		
	def PrepareModel(self, savedmodel = None):
		
		if savedmodel != None:
			self.knn = savedmodel
		else:
			if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION: 
				self.knn=neighbors.KNeighborsRegressor(self.n_neighbors, weights=self.weight)
			elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
				self.knn=neighbors.KNeighborsClassifier(self.n_neighbors, weights=self.weight)
			
			self.knn.fit(self.traindata ,self.trainlabel)
		
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.preprocess_method)
			

	def Predict(self):
		prediction=[]
		for testrecord in self.testdata :
			prediction.append( float(self.knn.predict(testrecord)[0]))
			
		self.result = [self.testlabel, prediction]
		
		
	def GetModel(self):
		return self.knn