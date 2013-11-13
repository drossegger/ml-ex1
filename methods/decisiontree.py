from sklearn import tree
from routines.preprocess import preprocess_apply
from base.algorithm import algorithmbase


class DecisionTree(algorithmbase):		
	
	def DoWork(self):
	
		traindata=preprocess_apply(self.traindata, self.preprocess_method)
		clf=tree.DecisionTreeRegressor()
		clf.fit(traindata,self.trainlabel)
		
		
		testdata=preprocess_apply(self.testdata, self.preprocess_method)
		prediction=[]
		for testrecord in testdata :
			prediction.append( clf.predict(testrecord))
			
		self.result = [self.testlabel, prediction]
