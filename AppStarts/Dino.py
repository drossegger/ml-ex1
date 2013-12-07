from routines.datareader import readCSVML
from routines.preprocess import preprocess_splitset
from methods.logisticregression import LogisticRegression
from base import algorithmcontainer
from base.constants import Constants
from methods.decisiontree import DecisionTree
from methods.neuralnetworkclassification import NeuralNetworkClassification
from methods.svm import SupportVectorMachine
from routines.preprocess import preprocess_splitset
import numpy as np
def Main():
  data=(readCSVML('data/ml-prove/all-data-raw.csv',range(0,53),range(53,58),','))
  #train=readCSVML('data/ml-prove/train.csv',range(0,51),range(51,57),',')
  
  labels=[]
  data[0]=[[101.0 if x==-100.0 else x for x in i] for i in data[0]]
  for x in data[0]:
    m=min(x)
    i
    if(m<1):
      i=1
    elif(m<10):
      i=2
    elif(m<25):
      i=3
    elif(m<50):
      i=4
    elif(m<100):
      i=5
    else:
      i=0
    labels.append(i)
  attrib=np.array(data[1])
  attrib=np.delete(attrib,4,1)
  attrib=np.delete(attrib,34,1)
  labels=np.asarray(labels)
  
    

  #newLabelsAll=[]
  #for x in [train,validation,test]:
  #  newLabels=[]
  #  for label in x[0]:
  #    for i in range(0,6):
  #      if(label[i]==1):
  #        label=(i+1)%6
  #        break
  #    newLabels.append(label)
  #  newLabelsAll.append(newLabels)
  #train[0],validation[0],test[0]=newLabelsAll[0],newLabelsAll[1],newLabelsAll[2]
  preprocessing_method=np.empty(51)
  preprocessing_method.fill(Constants.SCALING_METHOD_STANDARDIZATION)
  train,test=preprocess_splitset(attrib,labels)
  train,validation=preprocess_splitset(train[1],train[0])
  _container = algorithmcontainer.Container(train[1], train[0], test[1], test[0], Constants.MISSING_VALUE_METHOD_NONE, range(0,57), Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION,preprocessing_method)
  #_container.push(DecisionTree().SetAlgorithmName('DecisionTree'))
  _container.push(SupportVectorMachine().SetAlgorithmName('SVM_linear_1').ExtraParams(C=1,kernel='linear'))
  _container.push(SupportVectorMachine().SetAlgorithmName('SVM_linear_0.1').ExtraParams(C=0.1,kernel='linear'))
  _container.push(SupportVectorMachine().SetAlgorithmName('SVM_rbf_1').ExtraParams(C=1,kernel='rbf'))
  _container.push(SupportVectorMachine().SetAlgorithmName('SVM_rbf_0.1').ExtraParams(C=0.1,kernel='rbf'))
  _container.push(LogisticRegression().ExtraParams(C=0.5).SetAlgorithmName('LogisticRegression_0.5'))
  _container.push(LogisticRegression().ExtraParams(C=2).SetAlgorithmName('LogisticRegression_2'))
  _container.push(LogisticRegression().ExtraParams(C=5).SetAlgorithmName('LogisticRegression_5'))
    #
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=5, hiddenlayernodescount=51).SetAlgorithmName('NeuralNetwork_1_5'))
  _container.StartAlgorithms()
 

