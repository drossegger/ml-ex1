lrr <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/LinearRidgeRegression_MEAN_.1.csv',sep=';')
prr <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/PolynomialRidgeRegression_MEAN_32_.1.csv',sep=';')
nnr <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/NearestNeighborsRegression_MEAN_18.csv',sep=';')


	lrr__1 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116221852 - ZCore/LinearRidgeRegression_MEAN_.1.csv',sep=';')
	lrr__5 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116221852 - ZCore/LinearRidgeRegression_MEAN_.5.csv',sep=';')
	lrr_1 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116221852 - ZCore/LinearRidgeRegression_MEAN_1.csv',sep=';')
	lrr_10 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116221852 - ZCore/LinearRidgeRegression_MEAN_10.csv',sep=';')
	
	
	
	
	
	
	dataframe=data.frame(lrr__1$V3,lrr__5$V3,lrr_1$V3,lrr_10$V3)
	boxplot(dataframe,xlab='Alpha',ylab='Disparity from the actual consumption',xaxt='no')
	axis(1,at=c(1,2,3,4),labels=c('0.1', '0.5','1', '10'))
	savehistory('example.R')
	dev.copy(png,'C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/doc/src/figures/db3/lrralphavariatsZCore.png')
	dev.off()

	-------------------
	20131116220713 - MinMax
	20131116221852 - ZCore
	20131116214618 - NoNormalization
	-----------------
	
	nnr_2 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116220713 - MinMax/NearestNeighborsRegression_MEAN_10.csv',sep=';')
	nnr_5 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116221852 - ZCore/NearestNeighborsRegression_MEAN_10.csv',sep=';')
	nnr_10 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/NearestNeighborsRegression_MEAN_10.csv',sep=';')
	
	
	
	
	
	
	dataframe=data.frame(nnr_2$V3,nnr_5$V3,nnr_10$V3)
	boxplot(dataframe,xlab='Normalisation Method',ylab='Disparity from the actual consumption',xaxt='no')
	axis(1,at=c(1,2,3),labels=c('No normalisation', 'MaxMin','ZCore'))
	savehistory('example.R')
	dev.copy(png,'C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/doc/src/figures/db3/NNRCompare.png')
	dev.off()
	
	-----------------
	sgd_h_1 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/SGD_huber_.1.csv',sep=';')
	sgd_h_1000 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/SGD_huber_1000.csv',sep=';')
	sgd_s_1 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/SGD_squared_loss_.1.csv',sep=';')
	sgd_s_1000 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131116214618 - NoNormalization/SGD_squared_loss_1000.csv',sep=';')
	
	
	
	
	
	
	dataframe=data.frame(sgd_h_1$V3,sgd_h_1000$V3,sgd_s_1$V3,sgd_s_1000$V3)
	boxplot(dataframe,xlab='Loss (Epsilon)',ylab='Disparity from the actual consumption',xaxt='no')
	axis(1,at=c(1,2,3,4),labels=c('H(0.1)', 'H(1000)','SL(0.1)', 'SL(1000)'))
	savehistory('example.R')
	dev.copy(png,'C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/doc/src/figures/db3/SGDZcore.png')
	dev.off()
	
	-------------------
	
	
	nnr_2 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131117003918/NearestNeighborsRegression_MEAN_10.csv',sep=';')
	lrr__1 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131117003918/LinearRidgeRegression_MEAN__.1.csv',sep=';')
	sgd_s_1 <- read.table('C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/reports/20131117003918/SGD_squared_loss_.1.csv',sep=';')
	
	
	
	
	
	
	
	dataframe=data.frame(nnr_2$V3,lrr__1$V3,sgd_s_1$V3)
	boxplot(dataframe,xlab='Algorithm',ylab='Disparity from the actual consumption',xaxt='no')
	axis(1,at=c(1,2,3),labels=c('KNN', 'LRR','SGD'))
	savehistory('example.R')
	dev.copy(png,'C:/Users/morte_000/SkyDrive/Workspace/Study/TUWien Material/WS13, Machine Learning/Exercise/Ex01/ml-ex1/doc/src/figures/db3/FinalCompareLess.png')
	dev.off()
	
	
	
	