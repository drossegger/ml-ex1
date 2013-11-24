from base.algorithm import algorithmbase
from base.constants import Constants
import datetime
import sys
import time
import os
import cPickle

class Container(object):

    algorithms = []

    def __init__(self, traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames, mlmethod):
        self.traindata = traindata
        self.trainlabel = trainlabel
        self.testdata = testdata
        self.testlabel = testlabel
        self.preprocess_method = preprocess_method
        self.traincolumnnames = traincolumnnames
        self.mlmethod = mlmethod

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
    
    def StartAlgorithms(self, usesavedmodels=False):
        
        outputversion = 'reports/' + time.strftime("%Y%m%d%H%M%S", time.localtime())
        
        if self.mlmethod==Constants.MACHINE_LEARNING_METHOD_REGRESSION:
            finalresultcolumns = '{0:40}, {1:20}, {2:20}, {3:20}'.format('Algorithm', 'DiffMean', 'DiffStd', 'RunningTime')
        elif self.mlmethod==Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION:
            finalresultcolumns = '{0:40}'.format('Algorithm') + "\r\n"
            finalresultcolumns += '{0:20}, {1:20}, {2:20}, {3:20}'.format('Label', 'Prediction', 'Count', 'RunningTime')
        
        finalresultfile = self.MakeResultFile(outputversion, finalresultcolumns)
        
        print finalresultcolumns
        print ''
            
        algdirectory = 'algs/'    
        if not os.path.exists(algdirectory):
          os.makedirs(algdirectory)
        
        for _algorithm in self.algorithms:
            
            #try:
                
                _algorithm.Initiate(self.traindata, self.trainlabel, self.testdata, self.testlabel, self.preprocess_method, self.traincolumnnames, self.mlmethod)
                _algorithm.set_output_file_version(outputversion)
                _algorithm.set_final_output_file(finalresultfile)
                
                gnb_loaded = None
                if usesavedmodels:
                    try:
                        with open(algdirectory + _algorithm.algorithmlabel + '.pkl', 'rb') as fid:
                            gnb_loaded = cPickle.load(fid)
                            print _algorithm.algorithmlabel + ' loaded...'
                    except Exception, e:
                        print 'No model: ' + _algorithm.algorithmlabel
                
                
                _algorithm.PreProcessTrainData()
                _algorithm.StartFitting(gnb_loaded)
                
                try:
                    with open(algdirectory + _algorithm.algorithmlabel + '.pkl', 'wb') as fid:
                        _model = _algorithm.GetModel()
                        cPickle.dump(_model, fid)
                except Exception, e:
                    print 'Error in pickling : ' + str(e)
                
                _algorithm.PreProcessTestDate()
                _algorithm.Predict()
                
                _algorithm.print_output()
                _algorithm.print_output_file()
            
            #except Exception, e:
            #    print '----------------------------------------------------------------------'
            #    print 'Error in ' + _algorithm.algorithmlabel + ' : ' + str(e)
            #    print '----------------------------------------------------------------------'
                
        finalresultfile.close()
        print '...DONE...'
            