from routines.datareader import readCSV
from routines.polynomial import CreatePolynomial
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
    data=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc_train.data', range(0,9), 9,',')
    test=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc_test.data', range(0,9), 9,',')
 
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
    
    
    #creating polynomial data 
    '''
    numberofpolynomials=22
    traindata=CreatePolynomial(traindata, numberofpolynomials)
    testdata=CreatePolynomial(testdata, numberofpolynomials)
    '''
    
    _container = algorithmcontainer.Container(traindata, trainlabel, testdata, testlabel, missingvaluemethod, traincolumnnames, Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION,
                                              [Constants.SCALING_METHOD_STANDARDIZATION,
                                               Constants.SCALING_METHOD_STANDARDIZATION,
                                               Constants.SCALING_METHOD_STANDARDIZATION,
                                               Constants.SCALING_METHOD_STANDARDIZATION,
                                               Constants.SCALING_METHOD_STANDARDIZATION,
                                               Constants.SCALING_METHOD_STANDARDIZATION,
                                               Constants.SCALING_METHOD_STANDARDIZATION,
                                               Constants.SCALING_METHOD_STANDARDIZATION])
    
    #_container.push(Ridge().ExtraParams(alpha=.1).SetAlgorithmName('Ridge_.1'))
    _container.push(Ridge().ExtraParams(alpha=1).SetAlgorithmName('Ridge_1'))
    _container.push(RidgeCV().ExtraParams(alphas=[0.1, 1.0, 10.0]).SetAlgorithmName('RidgeCV_[0.1, 1.0, 10.0]'))
    #_container.push(LogisticRegression().ExtraParams(C=0.5).SetAlgorithmName('LogisticRegression_0.5'))
    #_container.push(LogisticRegression().ExtraParams(C=2).SetAlgorithmName('LogisticRegression_2'))
    _container.push(LogisticRegression().ExtraParams(C=5).SetAlgorithmName('LogisticRegression_5'))
    #_container.push(SGD().ExtraParams(loss='hinge').SetAlgorithmName('SGD_hinge'))
    #_container.push(SGD().ExtraParams(loss='modified_huber', epsilon=500).SetAlgorithmName('SGD_huber_1000'))
    _container.push(SGD().ExtraParams(loss='log').SetAlgorithmName('SGD_log'))
    _container.push(KNearestNeighbors().ExtraParams(n_neighbors=5, weight='uniform').SetAlgorithmName('KNearestNeighbors_5'))
    _container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=0.1).SetAlgorithmName('SupportVectorMachine_rbf_0.1'))
    _container.push(DecisionTree().SetAlgorithmName('DecisionTree'))
    #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=12).SetAlgorithmName('NeuralNetwork_1_5'))
    
    _container.StartAlgorithms()
    
    