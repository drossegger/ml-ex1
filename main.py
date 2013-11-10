'''
Created on 6 Nov 2013

@author: morte_000
'''
from methods.sgd import *
from methods.bayesianridgeregression import * 
from methods.ridgeregression import *
from methods.ridgeregressioncv import *
from methods.decisiontree import *
from methods.nearestneighborsregression import *
from methods.svm import *

from routines.preprocess import *
from routines.datareader import readCSV

def Main():
    data=readCSV('data/dataset1/auto-mpg.data', range(1,7), 0)
    test=readCSV('data/dataset1/auto-mpg-predictors.data', range(1,7), 0)
    
    traindata=data[1]
    trainlabel=data[0]
    testdata=test[1]
    testlabel=test[0]
    
    traindata=[[a if a!='?' else np.nan for a in instance] for instance in traindata]
    testdata=[[a if a!='?' else np.nan for a in instance] for instance in testdata]
    
    
    trainlabel=[float(a) for a in trainlabel]
    testlabel=[float(a) for a in testlabel]
        
    CalcBayesianRidgeRegression(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN)
    CalcRidgeRegression(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN, alpha=.5)
    CalcRidgeRegressionCV(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN, alphas=[0.1, 1.0, 10.0])
    CalcSGD(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN)
    CalcDecisionTree(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN)
    CalcNearestNeighborsRegression(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN, 
                                   n_neighbors=5, weight='uniform')#weight=uniform
    CalcSupportVectorMachine(traindata, trainlabel, testdata, testlabel, MISSING_VALUE_METHOD_MEAN);


if __name__ == '__main__':
    Main()