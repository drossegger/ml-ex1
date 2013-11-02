from sklearn import linear_model
import routines
import numpy as np

data=routines.readCSV('data/auto-mpg.data')
x=routines.preprocess(data[1])
clf = linear_model.BayesianRidge()
y=[float(a) for a in data[0]]
clf.fit(x,y )


testdata=routines.readCSV('data/auto-mpg-predictors.data')


testinputs=routines.preprocess(testdata[1])
prediction=[]
for testinput in testinputs :
	#predict mpg values and print prediction
	#print clf.predict([testinput[1],testinput[2]])
	prediction.append( clf.predict(testinput))
print prediction
print testdata[0]
diff=[float(a)-float(b) for a,b in zip(prediction,testdata[0])]
print diff
print np.mean(diff)
