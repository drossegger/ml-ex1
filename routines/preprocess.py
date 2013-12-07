from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedShuffleSplit
from base.constants import Constants
import numpy as np

def preprocess_apply(data, missingvaluemethod, preprocessingmethods):
	#imputing missing values
	if missingvaluemethod!=Constants.MISSING_VALUE_METHOD_NONE:
		if missingvaluemethod==Constants.MISSING_VALUE_METHOD_MEAN:
			imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
		elif missingvaluemethod==Constants.MISSING_VALUE_METHOD_MEDIAN:
			imp = Imputer(missing_values='NaN', strategy='median', axis=0)
		elif missingvaluemethod==Constants.MISSING_VALUE_METHOD_MOST_FREQUENT:
			imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
		imp.fit(data)
		data=imp.transform(data)
	else:
		data=np.asarray(data)

	#scale data
	res=np.array([])
	for i in range(0,len(preprocessingmethods)):
		field=[[x[i]] for x in data]
		if preprocessingmethods[i]==Constants.SCALING_METHOD_NONE:
			pass
		elif preprocessingmethods[i]==Constants.SCALING_METHOD_STANDARDIZATION:
			scaler=preprocessing.StandardScaler().fit(field)
			field=scaler.transform(field)
		elif preprocessingmethods[i]==Constants.SCALING_METHOD_MINMAX:
			field=preprocessing.MinMaxScaler().fit_transform(field)
		elif preprocessingmethods[i]==Constants.SCALING_METHOD_CATEGORICAL:
			enc = preprocessing.OneHotEncoder()
			enc.fit(field)
			field=enc.transform(field).toarray()
			
		if i==0:
			res = field
		else:
			res = np.concatenate((res, field), axis=1)
	return res
	
def preprocess_splitset(attributes,labels,validationsize=0.25):
	sss=StratifiedShuffleSplit(labels,1,test_size=validationsize)
	test_attrib,test_label,train_attrib,train_label=[],[],[],[]
	for train_i, test_i in sss:
		print len(train_i)
		for i in test_i:
			test_attrib.append(attributes[i])
			test_label.append(labels[i])
		for i in train_i:
			train_attrib.append(attributes[i])
			train_label.append(labels[i])
	return (np.asarray(train_label),np.asarray(train_attrib)),(np.asarray(test_label),np.asarray(test_attrib))
