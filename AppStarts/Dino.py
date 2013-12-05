from routines.datareader import readCSVML

def Main():
	train=readCSVML('data/ml-prove/train.csv',range(0,51),range(51,57),',')
	validation=readCSVML('data/ml-prove/validation.csv',range(0,51),range(51,57),',')
	test=readCSVML('data/ml-prove/test.csv',range(0,51),range(51,57),',')
	print "%s;%s"%(data[0][1],data[1][1])
	
