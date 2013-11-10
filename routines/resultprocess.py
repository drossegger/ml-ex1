import numpy as np

def printResult(result, label, labelColumnName):
	testlabel=result[0]
	prediction=result[1]
	diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
	diffmean=np.mean(diff)
	print label + ':' + '\r\n'
	for i in range(0, len(testlabel)):
		
		print '*******  Test instance: '+ str(i) +'  *******\r\n{0} : {1}\r\n	{2:10} ==> {3:10}\r\n	{4:10} ==> {5:10}\r\n'.format(labelColumnName,testlabel[i],'Prediction', prediction[i], 'Diff', diff[i])
		
	print '*******  TOTAL  *******'
	print ' ------------------------\r\n|{0:10} :: {1:10f}|\r\n ------------------------'.format('Diff Mean', diffmean)
	print '----------------------------------------------------'
