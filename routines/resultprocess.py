import numpy as np

def printResult(result, label):
	testlabel=result[0]
	prediction=result[1]
	diff=[float(a)-float(b) for a,b in zip(prediction,testlabel)]
	diffmean=np.mean(diff)
	print label + ':'
	print testlabel
	print prediction
	print diff
	print diffmean
	
