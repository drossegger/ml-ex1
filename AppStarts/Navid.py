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
     
    train=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/preproc2/cmc_train.data', range(1,22), 0,',')
    cv=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/preproc2/cmc_cv.data', range(1,22), 0,',')
    test=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/preproc2/cmc_test.data', range(1,22), 0,',')
 
    train_attr=train[1]
    train_label=train[0]
    train_label=[int(a-1) for a in train_label]
    train_column_names = train[2]
        
    cv_attr=cv[1]
    cv_label=cv[0]
    cv_label=[int(a-1) for a in cv_label]
    
    test_attr=test[1]
    test_label=test[0]
    test_label=[int(a-1) for a in test_label]
    #traindata=[[a if a!='?' else np.nan for a in instance] for instance in traindata]
    #testdata=[[a if a!='?' else np.nan for a in instance] for instance in testdata]
    
    
    missingvaluemethod = Constants.MISSING_VALUE_METHOD_NONE
    ispreprocessing=False
    preprocessingmethods = [];
                        
    #dimension reduction
    '''
    res = preprocess_apply(traindata, missingvaluemethod, preprocessingmethods)
    res = pca_apply(res, 2);
    res = np.concatenate((res, np.asarray([[x] for x in trainlabel])), axis=1)
    np.savetxt('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc.twodim', res, fmt='%.5f', delimiter=",")
    '''
    
    #splitting data
    '''
    data=readCSV('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc.data', range(0,9), 9,',')
    
    data_attr=data[1]
    data_label=data[0]
    data_label=[int(a) for a in data_label]
    data_column_names = data[2]
 
    data_attr=preprocess_apply(data_attr, missingvaluemethod, preprocessingmethods)
    
    [train_label,train_attrib],[test_label,test_attrib]=preprocess_splitset(data_attr,data_label,validationsize=0.20);
    testdata=np.append([[x] for x in test_label], test_attrib,axis=1)    
    np.savetxt('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/preproc1/cmc_test.data', testdata, fmt='%.4f', delimiter=",")
    [train_label,train_attrib],[cv_label,cv_attrib]=preprocess_splitset(train_attrib,train_label,validationsize=0.25);
    traindata=np.append([[x] for x in train_label], train_attrib,axis=1)
    cvdata=np.append([[x] for x in cv_label], cv_attrib,axis=1)
    np.savetxt('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/preproc1/cmc_train.data', traindata, fmt='%.4f', delimiter=",")
    np.savetxt('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/preproc1/cmc_cv.data', cvdata, fmt='%.4f', delimiter=",")
    '''
    
    #creating polynomial data 
    '''
    numberofpolynomials=22
    traindata=CreatePolynomial(traindata, numberofpolynomials)
    testdata=CreatePolynomial(testdata, numberofpolynomials)
    '''
    
    _container = algorithmcontainer.Container(train_attr, train_label, test_attr, test_label, missingvaluemethod, 
                                              train_column_names, Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION, 
                                              ispreprocessing, preprocessingmethods)
    
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
    #_container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=10).SetAlgorithmName('SupportVectorMachine_rbf_0.1'))
    #_container.push(DecisionTree().SetAlgorithmName('DecisionTree'))
    _container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=112).SetAlgorithmName('NeuralNetwork_1_5'))
    
    _container.StartAlgorithms()
    