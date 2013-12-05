from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection

from routines.preprocess import preprocess_apply
import numpy as np
from base.algorithm import algorithmbase
from base.constants import Constants

class  NeuralNetworkClassification(algorithmbase):		
	
	def ExtraParams(self, hiddenlayerscount, hiddenlayernodescount):
		self.hiddenlayerscount = hiddenlayerscount
		self.hiddenlayernodescount = hiddenlayernodescount
		return self
	
	def PreProcessTrainData(self):
		self.traindata = preprocess_apply(self.traindata, self.missingvaluemethod, self.preprocessingmethods)		
		
	def PrepareModel(self, savedmodel = None):
		
		if savedmodel != None:
			self.trainer = savedmodel
		else:
			attributescount=len(self.traindata[0])
			nrclass = len(set(self.trainlabel))
			self.ds = ClassificationDataSet(attributescount, target=nrclass, nb_classes=nrclass, class_labels=list(set(self.trainlabel)))
				
			for i in range(len(self.traindata)):
				self.ds.appendLinked(self.traindata[i], [self.trainlabel[i]])
			self.ds._convertToOneOfMany()
	
			self.net = FeedForwardNetwork()
			inLayer = LinearLayer(len(self.traindata[0]))
			self.net.addInputModule(inLayer)
			hiddenLayers=[]
			for i in range(self.hiddenlayerscount):
				hiddenLayer=SigmoidLayer(self.hiddenlayernodescount)
				hiddenLayers.append(hiddenLayer)
				self.net.addModule(hiddenLayer)
			outLayer = SoftmaxLayer(nrclass)
			self.net.addOutputModule(outLayer)
			
			layers_connections=[]
			layers_connections.append(FullConnection(inLayer, hiddenLayers[0]))
			for i in range(self.hiddenlayerscount-1):
				layers_connections.append(FullConnection(hiddenLayers[i-1], hiddenLayers[i]))
			layers_connections.append(FullConnection(hiddenLayers[-1], outLayer))
		
			for layers_connection in layers_connections:
				self.net.addConnection(layers_connection)
			self.net.sortModules()
			
			#training the network
			self.trainer = BackpropTrainer(self.net, self.ds)
			self.trainer.train()
		
		
	def PreProcessTestDate(self):
		self.testdata=preprocess_apply(self.testdata, self.missingvaluemethod, self.preprocessingmethods)
			

	def Predict(self):
		prediction=[]
		
		attributescount=len(self.testdata[0])
		nrclass = len(set(self.testlabel))
		dstraindata = ClassificationDataSet(attributescount, target=nrclass, nb_classes=nrclass, class_labels=list(set(self.testlabel)))
		for i in range(len(self.testdata)):
			dstraindata.appendLinked(self.testdata[i], self.testlabel[i])
		dstraindata._convertToOneOfMany()
		out = self.net.activateOnDataset(dstraindata)
		prediction = out.argmax(axis=1)
		'''
		for testrecord in self.testdata :
			out = self.net.activate(testrecord)[0]
			prediction.append(out)
		'''	
			
		self.result = 	[self.testlabel, prediction]
		
		
	def GetModel(self):
		return self.trainer