'''
Created on 13 Nov 2013

@author: morte_000
'''
from base.algorithm import algorithmbase

class Container(object):

    algorithms = []

    def __init__(self, traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames, labelindex):
        self.traindata = traindata
        self.trainlabel = trainlabel
        self.testdata = testdata
        self.testlabel = testlabel
        self.preprocess_method = preprocess_method
        self.traincolumnnames = traincolumnnames
        self.labelindex = labelindex
        

    def push(self, algorithm):
        self.algorithms.append(algorithm)
        return self    
        
        
    def StartAlgorithms(self):
        for _algorithm in self.algorithms:
            
            try:
                
                _algorithm.Initiate(self.traindata, self.trainlabel, self.testdata, self.testlabel, self.preprocess_method, self.traincolumnnames, self.labelindex)
                _algorithm.StartAlgorithm()
                _algorithm.print_output()
            
            except:
                print 'Error in ' + _algorithm.algorithmlabel
                
        print '...DONE...'
            