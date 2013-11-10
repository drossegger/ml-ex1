'''
Created on 6 Nov 2013

@author: morte_000
'''
from methods.sgd import *
from methods.bayesianridge import * 
from routines.preprocess import *

def Main():
    data=readCSV('data/auto-mpg.data', range(1,7), 0)
    test=readCSV('data/auto-mpg-predictors.data', range(1,7), 0)
    
    data_attributeList=data[1]
    data_label=data[0]
    test_attributeList=test[1]
    test_label=test[0]
    
    data_attributeList=[[a if a!='?' else np.nan for a in instance] for instance in data_attributeList]
    test_attributeList=[[a if a!='?' else np.nan for a in instance] for instance in test_attributeList]
    
    CalcBayesianRidge(data_attributeList, data_label, test_attributeList, test_label, MISSING_VALUE_METHOD_MEAN)
    CalcSGD(data_attributeList, data_label, test_attributeList, test_label, MISSING_VALUE_METHOD_MEAN)


if __name__ == '__main__':
    Main()