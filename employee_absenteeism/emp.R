#changing the working directory
setwd('D:/analytics/project')

#loading required packages
library(xlsx)
library(DMwR)
library(ggplot2)
library(corrplot)
library(rpart)
library(forecast)
library(randomForest)
library(caret)
library(class)

#reading the data
d=read.xlsx('emp_absent.xls',sheetIndex = 1)

str(d)
summary(d)

#variables
colnames(d)

#missing value analysis
mv=as.data.frame(apply(d,2,function(x){sum(is.na(x))}))
colnames(mv)[1]='count'
mv$percentage=(mv$count/nrow(d))*100

#missing value imputation
d=knnImputation(d,k=3)
d=round(d[,])

#verifying the data
apply(d,2,function(x){sum(is.na(x))})

#continuous variables
cont=c('ID','Transportation.expense','Distance.from.Residence.to.Work','Service.time','Age','Work.load.Average.day.','Hit.target',
      'Height','Weight','Body.mass.index')
#categorical variables
categ=c('Reason.for.absence','Month.of.absence','Day.of.the.week','Seasons','Disciplinary.failure','Education','Son','Social.drinker',
       'Social.smoker','Pet')

#outlier analysis
par(mfrow=c(2,5))
for (i in cont) {
  boxplot(d[,i],main=i)
}

#removing oultliers from 'Work.load.Average.day.' variable
fivenum(d$Work.load.Average.day.)
val=boxplot.stats(d$Work.load.Average.day.)$out
d$Work.load.Average.day.[d$Work.load.Average.day. %in% val]=NA

d=knnImputation(d,k=3)
boxplot(d$Work.load.Average.day.)

#Feature selection
#correlation analysis
corrplot(cor(d[,cont]),method = 'number')

#converting categorical variables to factor datatype
for(i in categ){
  print(i)
  d[,i]=as.factor(d[,i])
}

str(d)
hist(d$Absenteeism.time.in.hours)

#conducting Kruskal-Wallis test since target variable is continuous and not normally distributed
for (i in categ) {
  print(i)
  print(kruskal.test(d$Absenteeism.time.in.hours~d[,i]))
}


#'Weight' has a correlation of 0.9 with BMI.Hence removing weight from dataset
d=subset(d,select=-c(Weight))

#normality check
par(mfrow=c(2,5))
for(i in cont){
  hist(d[,i],main=i)
}


#Dividing data into train and test
ti=createDataPartition(d$Reason.for.absence,p=0.8,list = F)
train=d[ti,]
test=d[-ti,]

#Model development
#Decision tree
dt=rpart(Absenteeism.time.in.hours~.,data=train,method = 'anova')
p=predict(dt,test[,-20])
accuracy(test[,20],p)
#mape=54.10
#mae=4.27

#Random forest
rf=randomForest(Absenteeism.time.in.hours~.,data=train,ntree=500)
p=predict(rf,test[,-20])
accuracy(test[,20],p)
#mape=51.88
#mae=4.45

#linear regression
lm=lm(Absenteeism.time.in.hours~.,data=train)
summary(lm)
p=predict(lm,test[,-20])
accuracy(test[,20],p)
#mape=225.97
#mae=6.319

#KNN
km=knn(train[,-20],test[,-20],train[,20],k=5)
km=as.numeric(km)
accuracy(test[,20],km)
#mape=209.89
#mae=6.34

#Since Random forest has the lowest mape value, will freeze random forest model for this data set.