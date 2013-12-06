attach(mtcars)
par(mfrow=c(2,2))

table1<- read.table('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc_mixed.twodim', header=T, sep=',')
table11=table1[class==1,]
table12=table1[class==2,]
table13=table1[class==3,]
with(table11, plot(x, y, pch=20, col="red", main="Zero-mean, 1-N, None"))
with(table12, points(x, y, pch=20, col="black"))
with(table13, points(x, y, pch=20, col="green"))


table<- read.table('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc_standard.twodim', header=T, sep=',')
table1=table[class==1,]
table2=table[class==2,]
table3=table[class==3,]
with(table1, plot(x, y, pch=20, col="red", main="Zero-mean, 1-N, Zero-mean"))
with(table2, points(x, y, pch=20, col="black"))
with(table3, points(x, y, pch=20, col="green"))

table3<- read.table('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc_standard2.twodim', header=T, sep=',')
table31=table3[class==1,]
table32=table3[class==2,]
table33=table3[class==3,]
with(table31, plot(x, y, pch=20, col="red", main="Zero-mean, 1-N, 1-N"))
with(table32, points(x, y, pch=20, col="black"))
with(table33, points(x, y, pch=20, col="green"))

table2<- read.table('C:/Users/Navid/Documents/GitHub/ml-ex1/data/dataset2/cmc_minmax.twodim', header=T, sep=',')
table21=table2[class==1,]
table22=table2[class==2,]
table23=table2[class==3,]
with(table21, plot(x, y, pch=20, col="red", main="MinMax, 1-N, None"))
with(table22, points(x, y, pch=20, col="black"))
with(table23, points(x, y, pch=20, col="green"))


