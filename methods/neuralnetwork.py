from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection

from routines.preprocess import preprocess_apply
import numpy as np


def CalcNeuralNetwork(traindata, trainlabel, testdata, testlabel, preprocess_method, hiddenlayerscount, hiddenlayernodescount):
	attributescount=len(traindata[0])
	
	traindata=preprocess_apply(traindata, preprocess_method)
	ds = SupervisedDataSet(attributescount, 1)
	for i in range(len(traindata)):
		ds.appendLinked(traindata[i], trainlabel[i])
	
	#creating network structure
	net = FeedForwardNetwork()
	inLayer = LinearLayer(attributescount)
	net.addInputModule(inLayer)
	hiddenLayers=[]
	for i in range(hiddenlayerscount):
		hiddenLayer=SigmoidLayer(hiddenlayernodescount)
		hiddenLayers.append(hiddenLayer)
		net.addModule(hiddenLayer)
	outLayer = LinearLayer(1)
	net.addOutputModule(outLayer)

	layers_connections=[]
	layers_connections.append(FullConnection(inLayer, hiddenLayers[0]))
	for i in range(hiddenlayerscount-1):
		layers_connections.append(FullConnection(hiddenLayers[i-1], hiddenLayers[i]))
	layers_connections.append(FullConnection(hiddenLayers[-1], outLayer))

	for layers_connection in layers_connections:
		net.addConnection(layers_connection)
	net.sortModules()
	
	#training the network
	trainer = BackpropTrainer(net, ds)
	trainer.train()
	
	testdata=preprocess_apply(testdata, preprocess_method)
	prediction=[]
	for testrecord in testdata :
		prediction.append( net.activate(testrecord)[0])
		
	return [testlabel, prediction]
