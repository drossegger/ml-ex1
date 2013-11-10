def readCSV(path, attributeindexlist, labelindex):
	csvfile=open(path,'rb')
	y=[]
	x=[]
	lines=csvfile.read()
	csvfile.close()
	lines=lines.splitlines()
	for line in lines:
		line=line.split(';')
		y.append(line[labelindex])
		attributes=[]
		for c in attributeindexlist:
			attributes.append(line[c])
		x.append(attributes)
		#x.append()
	return [y,x]
