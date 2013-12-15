from routines.datareader import readCSVML
from routines.preprocess import preprocess_splitset
from methods.logisticregression import LogisticRegression
from base import algorithmcontainer
from base.constants import Constants
from methods.decisiontree import DecisionTree
from methods.neuralnetworkclassification import NeuralNetworkClassification
from methods.knearestneighbors import KNearestNeighbors
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
  preprocessing_method.fill(Constants.SCALING_METHOD_MINMAX)

  labelcount=[0,0,0,0,0,0]
  for label in labels:
    if label==0:
      labelcount[0]+=1
    elif label==1:
      labelcount[1]+=1
    elif label==2:
      labelcount[2]+=1
    elif label==3:
      labelcount[3]+=1
    elif label==4:
      labelcount[4]+=1
    elif label==5:
      labelcount[5]+=1
  print labelcount
  print [float(l)/float(sum(labelcount)) for l in labelcount]
  train,test=preprocess_splitset(attrib,labels)
  train,validation=preprocess_splitset(train[1],train[0])
  #_container = algorithmcontainer.Container(train[1], train[0], test[1],test[0], Constants.MISSING_VALUE_METHOD_NONE, range(0,57), Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION,False,preprocessing_method)
  #_container.push(LogisticRegression().ExtraParams(C=0.1).SetAlgorithmName('LogisticRegression_0.1'))
  #_container.push(LogisticRegression().ExtraParams(C=0.5).SetAlgorithmName('LogisticRegression_0.5'))
  #_container.push(LogisticRegression().ExtraParams(C=2).SetAlgorithmName('LogisticRegression_2'))
  #_container.push(LogisticRegression().ExtraParams(C=5).SetAlgorithmName('LogisticRegression_5'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=100).SetAlgorithmName('NeuralNetwork_1_300'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=125).SetAlgorithmName('NeuralNetwork_1_600'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=150).SetAlgorithmName('NeuralNetwork_1_900'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=1, hiddenlayernodescount=300).SetAlgorithmName('NeuralNetwork_1_1200'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=2, hiddenlayernodescount=300).SetAlgorithmName('NeuralNetwork_2_300'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=3, hiddenlayernodescount=300).SetAlgorithmName('NeuralNetwork_3_300'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=4, hiddenlayernodescount=300).SetAlgorithmName('NeuralNetwork_4_300'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=5, hiddenlayernodescount=300).SetAlgorithmName('NeuralNetwork_5_300'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=6, hiddenlayernodescount=300).SetAlgorithmName('NeuralNetwork_6_300'))
  #_container.push(NeuralNetworkClassification().ExtraParams(hiddenlayerscount=7, hiddenlayernodescount=300).SetAlgorithmName('NeuralNetwork_7_300'))
  #_container.push(DecisionTree().ExtraParams(criterion='gini').SetAlgorithmName('DecisionTree_gini'))
  #_container.push(DecisionTree().ExtraParams(criterion='entropy').SetAlgorithmName('DecisionTree_entropy'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='linear',C=0.1).SetAlgorithmName('SupportVectorMachine_linear_0.1'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='linear',C=1).SetAlgorithmName('SupportVectorMachine_linear_1'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='linear',C=5).SetAlgorithmName('SupportVectorMachine_linear_5'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='linear',C=10).SetAlgorithmName('SupportVectorMachine_linear_10'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='linear',C=20).SetAlgorithmName('SupportVectorMachine_linear_20'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='linear',C=50).SetAlgorithmName('SupportVectorMachine_linear_50'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='linear',C=100).SetAlgorithmName('SupportVectorMachine_linear_100'))
  #_container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=0.1).SetAlgorithmName('SupportVectorMachine_rbf_0.1'))
  #_container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=1).SetAlgorithmName('SupportVectorMachine_rbf_1'))
  #_container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=5).SetAlgorithmName('SupportVectorMachine_rbf_5'))
  #_container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=10).SetAlgorithmName('SupportVectorMachine_rbf_10'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=20).SetAlgorithmName('SupportVectorMachine_rbf_20'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=50).SetAlgorithmName('SupportVectorMachine_rbf_50'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='rbf',C=100).SetAlgorithmName('SupportVectorMachine_rbf_100'))
 ## _container.push(SupportVectorMachine().ExtraParams(kernel='poly',C=0.1).SetAlgorithmName('SupportVectorMachine_rbf_0.1'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='poly',C=1).SetAlgorithmName('SupportVectorMachine_rbf_1'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='poly',C=5).SetAlgorithmName('SupportVectorMachine_rbf_5'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='poly',C=10).SetAlgorithmName('SupportVectorMachine_rbf_10'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='poly',C=20).SetAlgorithmName('SupportVectorMachine_rbf_20'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='poly',C=50).SetAlgorithmName('SupportVectorMachine_rbf_50'))
 # _container.push(SupportVectorMachine().ExtraParams(kernel='poly',C=100).SetAlgorithmName('SupportVectorMachine_rbf_100'))
 # _container.push(LogisticRegression().ExtraParams(C=5).SetAlgorithmName('LogisticRegression_' + str(numberofpolynomials) + '_5'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=1, weight='uniform').SetAlgorithmName('KNearestNeighbors_1'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=5, weight='uniform').SetAlgorithmName('KNearestNeighbors_5'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=10, weight='uniform').SetAlgorithmName('KNearestNeighbors_10'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=20, weight='uniform').SetAlgorithmName('KNearestNeighbors_20'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=30, weight='uniform').SetAlgorithmName('KNearestNeighbors_30'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=40, weight='uniform').SetAlgorithmName('KNearestNeighbors_40'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=45, weight='uniform').SetAlgorithmName('KNearestNeighbors_45'))
 # _container.push(KNearestNeighbors().ExtraParams(n_neighbors=50, weight='uniform').SetAlgorithmName('KNearestNeighbors_50'))
 # _container.StartAlgorithms()
 

