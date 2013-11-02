import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

def readCSV(path):
	csvfile=open(path,'rb')
	y=[]
	x=[]
	instancename=[]
	lines=csvfile.read()
	csvfile.close()
	lines=lines.splitlines()
	for line in lines:
		line=line.split(';')
		y.append(line.pop(0))
		instancename.append(line.pop())
		x.append(line)
	return [y,x]

def preprocess(data):
	features=[[float(a) if a!='?' else '?' for a in instance] for instance in data]
	#replace missing data with mean
	nonmissingval=[instance[2] for instance in features if instance[2]!='?']
	mean=np.mean(nonmissingval)
	features=[[a if a!='?' else mean for a in instance] for instance in features]
	#scale features
	scaler=preprocessing.StandardScaler().fit(features)
	features=scaler.transform(features)
	return features


