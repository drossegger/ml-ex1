import numpy as np

def printResult(result, label):
	testlabel=result[0]
	prediction=result[1]
	diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
	diffmean=np.mean(diff)
	print label + ':' + '\r\n'
	for i in range(0, len(testlabel)):
		print '{0:10}\r\n	{1:10} ==> {2:10}\r\n	{3:10} ==> {4:10}'.format(testlabel[i],'Prediction', prediction[i], 'Diff', diff[i])
		
	print ' ------------------------\r\n|{0:10} :: {1:10f}|\r\n ------------------------'.format('Diff Mean', diffmean)
	print '----------------------------------------------------'
