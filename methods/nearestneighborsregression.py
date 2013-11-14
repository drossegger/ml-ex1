from sklearn import neighbors
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase


class NearestNeighborsRegression(algorithmbase):		
	
	def ExtraParams(self, n_neighbors, weight):
		self.n_neighbors = n_neighbors
		self.weight = weight
		return self
	
	
	
	def DoWork(self):
		
		self.traindata=preprocess_apply(self.traindata, self.preprocess_method)
		knn=neighbors.KNeighborsRegressor(self.n_neighbors, weights=self.weight)
		knn.fit(self.traindata,self.trainlabel)
		
		testdata=preprocess_apply(self.testdata, self.preprocess_method)
	
		prediction=[]
		for testrecord in testdata :
			prediction.append( knn.predict(testrecord))
			
		self.result = [self.testlabel, prediction]
