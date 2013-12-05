from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import StratifiedShuffleSplit
from base.constants import Constants
import numpy as np

def preprocess_apply(data, missingvaluemethod):
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
	
		#scale data
		scaler=preprocessing.StandardScaler().fit(data)
		data=scaler.transform(data)
	else:
		data=np.asarray(data)
	return data	
	
def preprocess_splitset(attributes,labels,validationsize=0.25):
	sss=StratifiedShuffleSplit(labels,1,test_size=validationsize)
	test_attrib,test_label,train_attrib,train_label
	for train_i, test_i in sss:
		test_attrib,test_label=attributes[test_i],labels[test_i]
		train_attrib,train_label=attributes[train_i],labels[train_i]
	return np.array([[train_label,train_attrib],[test_label,test_attrib]])
	



	

