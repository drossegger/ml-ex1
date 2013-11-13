'''
Created on 6 Nov 2013

@author: morte_000
'''

from routines.preprocess import *
from routines.datareader import readCSV
from routines.resultprocess import printResult

import numpy as np
from base import algorithmcontainer
from methods.bayesianridgeregression import BayesianRidgeRegression
from methods.ridgeregression import RidgeRegression
from methods.ridgeregressioncv import RidgeRegressionCV
from methods.sgd import SGD
from methods.nearestneighborsregression import NearestNeighborsRegression
from methods.neuralnetwork import NeuralNetwork
from methods.svm import SupportVectorMachine
from methods.decisiontree import DecisionTree


def Main():
    data=readCSV('data/dataset1/auto-mpg.data', range(1,7), 0)
    test=readCSV('data/dataset1/auto-mpg-predictors.data', range(1,7), 0)
    
    traindata=data[1]
    trainlabel=data[0]
    trainColumnNames = data[2]
    
    testdata=test[1]
    testlabel=test[0]
    testColumnNames = test[2]
    
    traindata=[[a if a!='?' else np.nan for a in instance] for instance in traindata]
    testdata=[[a if a!='?' else np.nan for a in instance] for instance in testdata]
    
    
    trainlabel=[float(a) for a in trainlabel]
    testlabel=[float(a) for a in testlabel]
    
    preprocess_method = MISSING_VALUE_METHOD_MEAN
    traincolumnnames = trainColumnNames
    labelindex = 0
    
    _container = algorithmcontainer.Container(traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames, labelindex)
    
    _container.push(BayesianRidgeRegression().SetAlgorithmName('BayesianRidgeRegression'))
    _container.push(RidgeRegression().ExtraParams(alpha=.5).SetAlgorithmName('RidgeRegression'))
    _container.push(RidgeRegressionCV().ExtraParams(alphas=[0.1, 1.0, 10.0]).SetAlgorithmName('RidgeRegressionCV'))
    _container.push(SGD().SetAlgorithmName('SGD'))
    _container.push(DecisionTree().SetAlgorithmName('DecisionTree'))
    _container.push(NearestNeighborsRegression().ExtraParams(n_neighbors=5, weight='uniform').SetAlgorithmName('NearestNeighborsRegression'))
    _container.push(SupportVectorMachine().SetAlgorithmName('SupportVectorMachine'))
    _container.push(NeuralNetwork().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=30).SetAlgorithmName('NeuralNetwork'))
    
    _container.StartAlgorithms()
    

if __name__ == '__main__':
    Main()