from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

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

data=readCSV('data/auto-mpg.data')

#convert all features to float, replace missing values by 0 (missing)
features=[[float(a) if a!='?' else 0 for a in instance] for instance in data[1]]

#scale features
clf = SGDClassifier(loss='hinge',shuffle=True,penalty='l2')
scaler=preprocessing.StandardScaler().fit(features)
features=scaler.transform(features)

#build model based on features 1 and 2 (horsepower and cylinders)
x=[[a[1],a[2]]for a in features]
clf.fit(x,data[0])


testdata=readCSV('data/auto-mpg-predictors.data')


testinputs=[[float(b) for b in a] for a in testdata[1]]
for testinput in testinputs :
	transformed=scaler.transform(testinput)
	#predict mpg values and print prediction
	print clf.predict([transformed[1],transformed[2]])

print testdata[0]
