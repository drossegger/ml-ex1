import numpy as np
import datetime
import os

class algorithmbase(object):

    result = []

    def Initiate(self, traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames, labelindex):
        self.traindata = traindata
        self.trainlabel = trainlabel
        self.testdata = testdata
        self.testlabel = testlabel
        self.preprocess_method = preprocess_method
        self.traincolumnnames = traincolumnnames
        self.labelindex = labelindex
        
    def SetAlgorithmName(self, algorithmlabel):
        self.algorithmlabel = algorithmlabel
        return self
    
    
    def StartAlgorithm(self):
        _algstart = datetime.datetime.utcnow()
        self.DoWork()
        _algend = datetime.datetime.utcnow()
        self.runningtime = _algend - _algstart
        
    
    def print_output(self):
        testlabel= self.result[0]
        prediction= self.result[1]
        diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
        diffmean =  np.mean(diff)
        print self.algorithmlabel + ':' + '\r\n'
        for i in range(0, len(testlabel)):
            print '*******  Test instance: '+ str(i) +'  *******\r\n{0} : {1}\r\n    {2:10} ==> {3:10}\r\n    {4:10} ==> {5:10}\r\n'.format(self.traincolumnnames[self.labelindex],testlabel[i],'Prediction', prediction[i], 'Diff', diff[i])
            
        print '*******  TOTAL  *******'
        print ' ------------------------\r\n|{0:10} :: {1:10f}|\r\n ------------------------'.format('Diff Mean', diffmean)
        print '----------------------------------------------------'
        
        
    def set_output_file_version(self, outputversion):
        self.outputversion = outputversion
        return self
    
    
    def print_output_file(self):
        if not os.path.exists(self.outputversion):
            os.makedirs(self.outputversion)
        
        file = open(self.outputversion + '/' + self.algorithmlabel  , 'w+')
        # TODO : writing report to the file
