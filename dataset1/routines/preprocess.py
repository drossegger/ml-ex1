import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

def preprocess_applymean(data, attributeindex):
	features=[[float(a) if a!='?' else '?' for a in instance] for instance in data]
	#replace missing data with mean
	nonmissingval=[instance[attributeindex] for instance in features if instance[attributeindex]!='?']
	mean=np.mean(nonmissingval)
	features=[[a if a!='?' else mean for a in instance] for instance in features]
	#scale features
	scaler=preprocessing.StandardScaler().fit(features)
	features=scaler.transform(features)
	return features


