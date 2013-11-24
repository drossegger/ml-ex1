from sklearn import preprocessing
from sklearn.preprocessing import Imputer
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
	