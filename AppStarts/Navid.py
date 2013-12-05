import numpy as np
from routines.datareader import readCSV
from routines.polynomial import CreatePolynomial
from routines.preprocess import preprocess_apply
from routines.preprocess import preprocess_splitset
from routines.dimensionreduction import pca_apply
from base import algorithmcontainer
from base.constants import Constants
from methods.ridge import Ridge
from methods.neuralnetworkclassification import NeuralNetworkClassification
from methods.neuralnetworkregression import NeuralNetworkRegression
from methods.ridgecv import RidgeCV
from methods.logisticregression import LogisticRegression
from methods.sgd import SGD
from methods.svm import SupportVectorMachine
from methods.knearestneighbors import KNearestNeighbors
from methods.decisiontree import DecisionTree

def Main():
    data=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc.data', range(0,9), 9,',')
    test=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc.data', range(0,9), 9,',')
 
    traindata=data[1]
    trainlabel=data[0]
    trainColumnNames = data[2]
    
    testdata=test[1]
    testlabel=test[0]
    testColumnNames = test[2]
    
    #traindata=[[a if a!='?' else np.nan for a in instance] for instance in traindata]
    #testdata=[[a if a!='?' else np.nan for a in instance] for instance in testdata]
    
    missingvaluemethod = Constants.MISSING_VALUE_METHOD_NONE
    traincolumnnames = trainColumnNames
    preprocessingmethods = [Constants.SCALING_METHOD_STANDARDIZATION,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_STANDARDIZATION,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL];
    '''
    preprocessingmethods = [Constants.SCALING_METHOD_STANDARDIZATION,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_STANDARDIZATION,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL,
                           Constants.SCALING_METHOD_CATEGORICAL];
    '''                       
    #dimension reduction
    '''
    res = preprocess_apply(traindata, missingvaluemethod, preprocessingmethods)
    res = pca_apply(res, 2);
    res = np.concatenate((res, np.asarray([[x] for x in trainlabel])), axis=1)
    np.savetxt('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc.twodim', res, fmt='%.5f', delimiter=",")
    '''
    
    traindata,trainlabel=preprocess_splitset(traindata,trainlabel,validationsize=0.25);
    
    #creating polynomial data 
    '''
    numberofpolynomials=22
    traindata=CreatePolynomial(traindata, numberofpolynomials)
    testdata=CreatePolynomial(testdata, numberofpolynomials)
    '''
    
    _container = algorithmcontainer.Container(traindata, trainlabel, testdata, testlabel, missingvaluemethod, 
                                              traincolumnnames, Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION, preprocessingmethods)
    
    #_container.push(Ridge().ExtraParams(alpha=.1).SetAlgorithmName('Ridge_.1'))
    #_container.push(Ridge().ExtraParams(alpha=1).SetAlgorithmName('Ridge_1'))
    #_container.push(RidgeCV().ExtraParams(alphas=[0.1, 1.0, 10.0]).SetAlgorithmName('RidgeCV_[0.1, 1.0, 10.0]'))
    #_container.push(LogisticRegression().ExtraParams(C=0.5).SetAlgorithmName('LogisticRegression_0.5'))
    #_container.push(LogisticRegression().ExtraParams(C=2).SetAlgorithmName('LogisticRegression_2'))
    #_container.push(LogisticRegression().ExtraParams(C=5).SetAlgorithmName('LogisticRegression_5'))
    #_container.push(SGD().ExtraParams(loss='hinge').SetAlgorithmName('SGD_hinge'))
    #_container.push(SGD().ExtraParams(loss='modified_huber', epsilon=500).SetAlgorithmName('SGD_huber_1000'))
    #_container.push(SGD().ExtraParams(loss='log').SetAlgorithmName('SGD_log'))
    #_container.push(KNearestNeighbors().ExtraParams(n_neighbors=5, weight='uniform').SetAlgorithmName('KNearestNeighbors_5'))
    #_container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=0.1).SetAlgorithmName('SupportVectorMachine_rbf_0.1'))
    #_container.push(DecisionTree().SetAlgorithmName('DecisionTree'))
    _container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=12).SetAlgorithmName('NeuralNetwork_1_5'))
    
    _container.StartAlgorithms()
    
    