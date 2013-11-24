from routines.datareader import readCSV
from routines.polynomial import CreatePolynomial
from base import algorithmcontainer
from base.constants import Constants
from methods.linearridgeregression import LinearRidgeRegression
from methods.sgd import SGD
from methods.svm import SupportVectorMachine
from methods.knearestneighbors import KNearestNeighbors


def Main():
    data=readCSV('C:/Users/Navid/Documents/GitHub/vectorized-music/app/resource/tokens/english/gutenberg/books17_train', range(1,7), 7,';')
    test=readCSV('C:/Users/Navid/Documents/GitHub/vectorized-music/app/resource/tokens/english/gutenberg/books17_test', range(1,7), 7,';')
 
    traindata=data[1]
    trainlabel=data[0]
    trainColumnNames = data[2]
    
    testdata=test[1]
    testlabel=test[0]
    testColumnNames = test[2]
    
    #traindata=[[a if a!='?' else np.nan for a in instance] for instance in traindata]
    #testdata=[[a if a!='?' else np.nan for a in instance] for instance in testdata]
    
    preprocess_method = Constants.MISSING_VALUE_METHOD_NONE
    traincolumnnames = trainColumnNames
    
    
    #creating polynomial data 
    '''
    numberofpolynomials=22
    traindata=CreatePolynomial(traindata, numberofpolynomials)
    testdata=CreatePolynomial(testdata, numberofpolynomials)
    '''
    
    _container = algorithmcontainer.Container(traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames, Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION)
    
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=.1).SetAlgorithmName('PolynomialRidgeRegression_MEAN_' + str(numberofpolynomials) + '_.1'))
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=.5).SetAlgorithmName('PolynomialRidgeRegression_MEAN_' + str(numberofpolynomials) + '_.5'))
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=1).SetAlgorithmName('PolynomialRidgeRegression_MEAN_' + str(numberofpolynomials) + '_1'))
    #_container.push(LinearRidgeRegression().ExtraParams(alpha=10).SetAlgorithmName('PolynomialRidgeRegression_MEAN_' + str(numberofpolynomials) + '_10'))
    #_container.push(LinearRidgeRegressionCV().ExtraParams(alphas=[0.1, 1.0, 10.0]).SetAlgorithmName('PolynomialRidgeRegression_MEAN_' + str(numberofpolynomials)))
    #_container.push(LinearBayesianRidgeRegression().SetAlgorithmName('LinearBayesianRidgeRegression_MEAN'))
    #_container.push(KNearestNeighbors().ExtraParams(n_neighbors=5, weight='uniform').SetAlgorithmName('KNearestNeighbors_5'))
    _container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=0.1).SetAlgorithmName('SupportVectorMachine_rbf_0.1'))
    #_container.push(SGD().ExtraParams(loss='huber', epsilon=1000).SetAlgorithmName('SGD_huber_1000'))
    #_container.push(NeuralNetwork().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=5).SetAlgorithmName('NeuralNetwork_1_5'))
    
    _container.StartAlgorithms()
    
    