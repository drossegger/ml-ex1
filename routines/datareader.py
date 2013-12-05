def readCSV(path, attributeindexlist, labelindex, delim):
	csvfile=open(path,'rb')
	y=[]
	x=[]
	z=[]
	lines=csvfile.read()
	csvfile.close()
	lines=lines.splitlines()
	line = lines[0]
	line=line.split(delim)
	z= list(columnName for columnName in line)
	for line in lines[1:]:

	 	if line.strip() == '':
	 		continue
	 	
		line=line.split(delim)

		
		if line[labelindex].strip() == 'NaN':
			continue
		
		y.append(float(line[labelindex]))
		attributes=[]
		for c in attributeindexlist:
			attributes.append(float(line[c]))
		x.append(attributes)
		#x.append()
	return [y,x,z]

def readCSV_dsProve(path, attributeindexlist, labelindexlist, delim):
	csvfile=open(path,'rb')
	x,y=[],[]
	lines=csvfile.read()
	csvfile.close()
	lines=lines.splitlines()
	for line in lines:
		line=line.split(delim)
		labels=[]
		for c in labelindexlist:
			labels.append(int(line[c]))
		y.append(labels)
		attributes=[]
		for c in attributeindexlist:
			attributes.append(float(line[c]))
		x.append(attributes)
	return [y,x]

		

