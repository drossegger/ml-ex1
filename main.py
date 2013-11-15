'''
Created on 6 Nov 2013

@author: morte_000
'''

from routines.preprocess import *
from routines.datareader import readCSV
from routines.resultprocess import printResult

import numpy as np
from base import algorithmcontainer
from methods._linearbayesianridgeregression import LinearBayesianRidgeRegression
from methods.linearridgeregression import LinearRidgeRegression
from methods._linearridgeregressioncv import LinearRidgeRegressionCV
from methods.sgd import SGD
from methods.nearestneighborsregression import NearestNeighborsRegression
from methods._neuralnetwork import NeuralNetwork
from methods.svm import SupportVectorMachine
from methods._decisiontree import DecisionTree


def Main():
    data=readCSV('data/dataset1/auto-mpg.data', range(1,8), 0,';')
    test=readCSV('data/dataset1/auto-mpg-predictors.data', range(1,8), 0,';')
    
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
    
    preprocess_method = MISSING_VALUE_METHOD_MOST_FREQUENT
    traincolumnnames = trainColumnNames
    labelindex = 0
    
    _container = algorithmcontainer.Container(traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames, labelindex)
    
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=.1).SetAlgorithmName('LinearRidgeRegression_MOST_FREQUENT_.1'))
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=.5).SetAlgorithmName('LinearRidgeRegression_MOST_FREQUENT_.5'))
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=1).SetAlgorithmName('LinearRidgeRegression_MOST_FREQUENT_1'))
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=10).SetAlgorithmName('LinearRidgeRegression_MOST_FREQUENT_10'))
    #_container.push(LinearRidgeRegressionCV().ExtraParams(alphas=[0.1, 1.0, 10.0]).SetAlgorithmName('LinearRidgeRegressionCV_MOST_FREQUENT'))
    #_container.push(LinearBayesianRidgeRegression().SetAlgorithmName('LinearBayesianRidgeRegression_MOST_FREQUENT'))
    #_container.push(SGD().ExtraParams(loss='hinge').SetAlgorithmName('SGD'))
    #_container.push(DecisionTree().SetAlgorithmName('DecisionTree'))
    #_container.push(NearestNeighborsRegression().ExtraParams(n_neighbors=5, weight='uniform').SetAlgorithmName('NearestNeighborsRegression'))
    #_container.push(SupportVectorMachine().SetAlgorithmName('SupportVectorMachine'))
    #_container.push(NeuralNetwork().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=30).SetAlgorithmName('NeuralNetwork'))
    
    _container.StartAlgorithms()
    

if __name__ == '__main__':
    Main()
