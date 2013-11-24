from routines.datareader import readCSV
from routines.polynomial import CreatePolynomial
from base import algorithmcontainer
from methods.linearridgeregression import LinearRidgeRegression
from methods._linearbayesianridgeregression import LinearBayesianRidgeRegression
from methods.sgdregression import SGDRegression
from methods.nearestneighborsregression import NearestNeighborsRegression


def Main():
    data=readCSV('data/dataset3/household_power_consumption_preprocessed.txt', range(0,5), 6,';')
    test=readCSV('data/dataset3/household_power_consumption_test_preprocessed.txt', range(0,5), 6,';')
 
    traindata=data[1]
    trainlabel=data[0]
    trainColumnNames = data[2]
    
    testdata=test[1]
    testlabel=test[0]
    testColumnNames = test[2]
    
    #traindata=[[a if a!='?' else np.nan for a in instance] for instance in traindata]
    #testdata=[[a if a!='?' else np.nan for a in instance] for instance in testdata]
    
    
    trainlabel=[float(a) for a in trainlabel]
    testlabel=[float(a) for a in testlabel]
    
    preprocess_method = MISSING_VALUE_METHOD_MEAN
    traincolumnnames = trainColumnNames
    
    
    #creating polynomial data 
    numberofpolynomials=''
    #traindata=CreatePolynomial(traindata, numberofpolynomials)
    #testdata=CreatePolynomial(testdata, numberofpolynomials)
    
    
    _container = algorithmcontainer.Container(traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames)
    
    _container.push(LinearRidgeRegression().ExtraParams(alpha=.1).SetAlgorithmName('LinearRidgeRegression_MEAN_' + str(numberofpolynomials) + '_.1'))
    _container.push(LinearRidgeRegression().ExtraParams(alpha=.5).SetAlgorithmName('LinearRidgeRegression_MEAN_' + str(numberofpolynomials) + '_.5'))
    _container.push(LinearRidgeRegression().ExtraParams(alpha=1).SetAlgorithmName('LinearRidgeRegression_MEAN_' + str(numberofpolynomials) + '_1'))
    _container.push(LinearRidgeRegression().ExtraParams(alpha=10).SetAlgorithmName('LinearRidgeRegression_MEAN_' + str(numberofpolynomials) + '_10'))
    #_container.push(LinearRidgeRegressionCV().ExtraParams(alphas=[0.1, 1.0, 10.0]).SetAlgorithmName('PolynomialRidgeRegression_MEAN_' + str(numberofpolynomials)))
    #_container.push(LinearBayesianRidgeRegression().SetAlgorithmName('LinearBayesianRidgeRegression_MEAN'))
    _container.push(NearestNeighborsRegression().ExtraParams(n_neighbors=2, weight='uniform').SetAlgorithmName('NearestNeighborsRegression_MEAN_2'))
    _container.push(NearestNeighborsRegression().ExtraParams(n_neighbors=5, weight='uniform').SetAlgorithmName('NearestNeighborsRegression_MEAN_5'))
    _container.push(NearestNeighborsRegression().ExtraParams(n_neighbors=10, weight='uniform').SetAlgorithmName('NearestNeighborsRegression_MEAN_10'))
    #_container.push(SupportVectorMachine().SetAlgorithmName('SupportVectorMachine_MEAN'))
    _container.push(SGD().ExtraParams(loss='huber',epsilon=0.1).SetAlgorithmName('SGD_huber_.1'))
    _container.push(SGD().ExtraParams(loss='squared_loss',epsilon=0.1).SetAlgorithmName('SGD_squared_loss_.1'))
    _container.push(SGD().ExtraParams(loss='huber',epsilon=1000).SetAlgorithmName('SGD_huber_1000'))
    _container.push(SGD().ExtraParams(loss='squared_loss',epsilon=1000).SetAlgorithmName('SGD_squared_loss_1000'))
    #_container.push(NeuralNetwork().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=5).SetAlgorithmName('NeuralNetwork_1_5'))
    
    _container.StartAlgorithms()
    
    