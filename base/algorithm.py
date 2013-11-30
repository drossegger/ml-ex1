import numpy as np
import datetime
import os
from base.constants import Constants
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class algorithmbase(object):

    result = []

    
    def Initiate(self, traindata, trainlabel, testdata, testlabel, preprocess_method, traincolumnnames, mlmethod):
        self.traindata = traindata
        self.trainlabel = trainlabel
        self.testdata = testdata
        self.testlabel = testlabel
        self.preprocess_method = preprocess_method
        self.traincolumnnames = traincolumnnames
        self.mlmethod = mlmethod;
        
    def SetAlgorithmName(self, algorithmlabel):
        self.algorithmlabel = algorithmlabel
        return self
    
    
    def StartFitting(self, savedmodel=None):
        _algstart = datetime.datetime.utcnow()
        self.PrepareModel(savedmodel)
        _algend = datetime.datetime.utcnow()
        self.runningtime = _algend - _algstart
    
    def print_output(self):
        testlabel= self.result[0]
        prediction= self.result[1]
        testlabel_prediction=zip(testlabel,prediction)
            
        if (self.mlmethod == Constants.MACHINE_LEARNING_METHOD_REGRESSION):
            diff=[float(b)-float(a) for a,b in testlabel_prediction]
            diffmean =  np.mean(diff)
            diffstd =  np.std(diff)
            output = '{0:<40}, {1:<20}, {2:<20}, {3:<20}'.format(self.algorithmlabel, diffmean, diffstd, self.runningtime) 
            print  output
            self.finaloutputfile.write(output + '\n')
        elif (self.mlmethod == Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION):
            output=self.algorithmlabel
            self.finaloutputfile.write(output + '\n')
            print  output
            testlabel_prediction_distinct=list(set(testlabel_prediction))
            for i in range(0,len(testlabel_prediction_distinct)):
                output = '{0:<20}, {1:<20}, {2:<20}, {3:<20}'.format(testlabel_prediction_distinct[i][0], testlabel_prediction_distinct[i][1], testlabel_prediction.count(testlabel_prediction_distinct[i]), self.runningtime) 
                print  output
                self.finaloutputfile.write(output + '\n')
            output = 'precision:{0:<20}'.format(precision_score(testlabel, prediction, average='micro'))
            print  output
            self.finaloutputfile.write(output + '\n')
            output = 'recall:{0:<20}'.format(recall_score(testlabel, prediction, average='micro'))
            print  output
            self.finaloutputfile.write(output + '\n')
        self.finaloutputfile.flush()
        
    def set_output_file_version(self, outputversion):
        self.outputversion = str(outputversion)
        return self
    
    def set_final_output_file(self, finaloutputfile):
        self.finaloutputfile = finaloutputfile
        return self
    
    def print_output_file(self):
        if not os.path.exists(self.outputversion):
          os.makedirs(self.outputversion)
        file = open(self.outputversion + '/' + self.algorithmlabel + '.csv'  , 'w+')
        
        testlabel= self.result[0]
        prediction= self.result[1]
        testlabel_prediction=zip(testlabel,prediction)
        
        if (self.mlmethod == Constants.MACHINE_LEARNING_METHOD_REGRESSION):
            diff=[float(b)-float(a) for a,b in testlabel_prediction]
            for i in range(0,len(testlabel)):
                file.write( '%s;%s;%s\n'%(testlabel[i],prediction[i],diff[i]) )
        elif (self.mlmethod == Constants.MACHINE_LEARNING_METHOD_CLASSIFICATION):
            testlabel_prediction_distinct=list(set(testlabel_prediction))
            for i in range(0,len(testlabel_prediction_distinct)):
                file.write('%s;%s;%s\n'%(testlabel_prediction_distinct[i][0], testlabel_prediction_distinct[i][1], testlabel_prediction.count(testlabel_prediction_distinct[i])))
        file.close()
        return self
