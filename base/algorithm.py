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
        #self.print_output_file()
        testlabel= self.result[0]
        prediction= self.result[1]
        diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
        diffmean =  np.mean(diff)
        
        print  '{0:<40}, {1:<20}, {2:<20}'.format(self.algorithmlabel , diffmean,  self.runningtime)
            
        
        
    def set_output_file_version(self, outputversion):
        self.outputversion = str(outputversion)
        return self
    
    
    def print_output_file(self):
        if not os.path.exists(self.outputversion):
          os.makedirs(self.outputversion)
        file = open(self.outputversion + '/' + self.algorithmlabel  , 'w+')
        testlabel=self.result[0]
        prediction=self.result[1]
        diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
        for i in range(0,len(testlabel)):
            file.write( '%s;%s;%s\n'%(testlabel[i],prediction[i],diff[i]) )
        file.close()
        return self
