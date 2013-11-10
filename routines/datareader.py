def readCSV(path, attributeindexlist, labelindex):
	csvfile=open(path,'rb')
	y=[]
	x=[]
	z=[]
	lines=csvfile.read()
	csvfile.close()
	lines=lines.splitlines()
	line = lines[0]
	line=line.split(';')
	z= list(columnName for columnName in line)
	for line in lines[1:]:
		line=line.split(';')
		y.append(line[labelindex])
		attributes=[]
		for c in attributeindexlist:
			attributes.append(line[c])
		x.append(attributes)
		#x.append()
	return [y,x,z]
