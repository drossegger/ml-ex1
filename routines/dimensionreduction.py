from sklearn import decomposition
from base.constants import Constants
import numpy as np

def pca_apply(data, n_components):
	
	pca = decomposition.PCA(n_components=n_components)
	pca.fit(data)
	res=pca.transform(data)
	return res
	