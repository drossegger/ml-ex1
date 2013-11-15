from sklearn import neighbors
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase


class NearestNeighborsRegression(algorithmbase):		
	
	def ExtraParams(self, n_neighbors, weight):
		self.n_neighbors = n_neighbors
		self.weight = weight
		return self
	
	
	
	def DoWork(self):
		'''
		Deprecated
		'''
		
		self.traindata=preprocess_apply(self.traindata, self.preprocess_method)
		knn=neighbors.KNeighborsRegressor(self.n_neighbors, weights=self.weight)
		knn.fit(self.traindata,self.trainlabel)
		
		
		testdata=preprocess_apply(self.testdata, self.preprocess_method)
	
		prediction=[]
		for testrecord in testdata :
			prediction.append( knn.predict(testrecord)[0])
			
		self.result = [self.testlabel, prediction]




	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.preprocess_method)
		
		
	def PrepareModel(self, savedmodel = None):
		
		if savedmodel != None:
			self.knn = savedmodel
		else:
			self.knn=neighbors.KNeighborsRegressor(self.n_neighbors, weights=self.weight)
			self.knn.fit(self.traindata ,self.trainlabel)
		
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.preprocess_method)
			

	def Predict(self):
		prediction=[]
		for testrecord in self.testdata :
			prediction.append( self.knn.predict(testrecord)[0])
			
		self.result = 	[self.testlabel, prediction]
		
		
	def GetModel(self):
		return self.knn