def readCSV(path, attributeindexlist, labelindex,delim):
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
<<<<<<< SplittedTasks
	 	if line.strip() == '':
	 		continue
		line=line.split(delim)
=======
		
		if line.strip() == '':
			continue
		
		line=line.split(delim)
		
>>>>>>> local
		if line[labelindex].strip() == 'NaN':
			continue
		
		y.append(line[labelindex])
		attributes=[]
		for c in attributeindexlist:
			attributes.append(line[c])
		x.append(attributes)
		#x.append()
	return [y,x,z]
