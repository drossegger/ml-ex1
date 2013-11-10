'''
Created on 6 Nov 2013

@author: morte_000
'''
from methods.sgd import CalcSGD
from methods.bayesianridgeregression import CalcBayesianRidgeRegression 
from methods.ridgeregression import CalcRidgeRegression
from methods.ridgeregressioncv import CalcRidgeRegressionCV
from methods.decisiontree import CalcDecisionTree
from methods.nearestneighborsregression import CalcNearestNeighborsRegression
from methods.svm import CalcSupportVectorMachine
from methods.neuralnetwork import CalcNeuralNetwork

from routines.preprocess import *
from routines.datareader import readCSV
from routines.resultprocess import printResult

import numpy as np

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
    
    printResult(CalcBayesianRidgeRegression(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN), 'Bayesian Ridge Regression', trainColumnNames[0])
    printResult(CalcRidgeRegression(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN, alpha=.5), 'Ridge Regression', trainColumnNames[0])
    printResult(CalcRidgeRegressionCV(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN, alphas=[0.1, 1.0, 10.0]), 'Ridge Regression CV', trainColumnNames[0])
    printResult(CalcSGD(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN), 'Stochastic Gradient Descent', trainColumnNames[0])
    printResult(CalcDecisionTree(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN), 'Decision Tree', trainColumnNames[0])
    printResult(CalcNearestNeighborsRegression(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN, 
                                   n_neighbors=5, weight='uniform'), 'Nearest Neighbors Regression', trainColumnNames[0])#weight=uniform
    printResult(CalcSupportVectorMachine(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN), 'Support Vector Machine Regression', trainColumnNames[0])
    printResult(CalcNeuralNetwork(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN, 
                                  hiddenlayerscount=1, hiddenlayernodescount=30), 'Neural Network', trainColumnNames[0])
    


if __name__ == '__main__':
    Main()