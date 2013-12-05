from routines.datareader import readCSV_dsProve

def Main():
	train=readCSV_dsProve('data/ml-prove/train.csv',range(0,51),range(51,57),',')
	validation=readCSV_dsProve('data/ml-prove/validation.csv',range(0,51),range(51,57),',')
	test=readCSV_dsProve('data/ml-prove/test.csv',range(0,51),range(51,57),',')
	
