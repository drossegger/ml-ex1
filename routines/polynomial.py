def CreatePolynomial(data, numberofpolynomials):
	for row in data[:]:
		attributes=row[:]
		for p in range(2, numberofpolynomials+1):
			for val in attributes:
				row.append(str(pow(float(val),p)))
	return data
