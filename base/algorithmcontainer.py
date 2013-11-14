'''
Created on 13 Nov 2013

@author: morte_000
'''
from base.algorithm import algorithmbase
import datetime
import sys
import time
import os

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
        
        
    def MakeResultFile(self, outputversion, finalresultcolumns):
        if not os.path.exists(outputversion):
          os.makedirs(outputversion)
        finalresultfile = open(outputversion + '/' + 'FinalResult' + '.csv'  , 'w+')
        finalresultfile.write(finalresultcolumns + '\n')
        finalresultfile.flush()
        return finalresultfile
    
    def StartAlgorithms(self):
        outputversion = 'reports/' + time.strftime("%Y%m%d%H%M%S", time.localtime())
        
        finalresultcolumns = '{0:40}, {1:20}, {2:20}'.format('Algorithm' , 'DiffMean',  'RunningTime')
        
        finalresultfile = self.MakeResultFile(outputversion, finalresultcolumns)
        
        print finalresultcolumns
        print ''
            
        
        for _algorithm in self.algorithms:
            
            try:
                
                _algorithm.Initiate(self.traindata, self.trainlabel, self.testdata, self.testlabel, self.preprocess_method, self.traincolumnnames, self.labelindex)
                _algorithm.set_output_file_version(outputversion)
                _algorithm.set_final_output_file(finalresultfile)
                _algorithm.StartAlgorithm()
                _algorithm.print_output()
                _algorithm.print_output_file()
            
            except Exception, e:
                print '----------------------------------------------------------------------'
                print 'Error in ' + _algorithm.algorithmlabel + ' : ' + str(e)
                print '----------------------------------------------------------------------'
                
        finalresultfile.close()
        print '...DONE...'
            