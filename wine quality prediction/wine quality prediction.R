rm(list = ls())

#loading libraries
library(DMwR)
library(caret)
library(corrplot)
library(forecast)
library(rpart)
library(randomForest)
library(gbm)
library(xgboost)
library(class)
library(e1071)

#importing data from csv to a dataframe
d=read.csv('D:/analytics/practice/wine.csv')

#checking the datatypes of all variables
str(d)

summary(d)

#Missing value analysis
apply(d,2,function(x){sum(is.na(x))})

#exploratory data analysis
dp=aggregate.data.frame(d,by =list(d$quality),mean)

par(mfrow=c(3,4))
for(i in colnames(dp)[-12]){
  barplot(dp[,i],dp$quality,xlab = 'quality',ylab = i)
}

#outlier analysis
par(mfrow=c(3,4))
for(i in colnames(d)[-12]){
  boxplot(d[,i],main=i)
}

#feature selection
corrplot(cor(d),method = 'number')

#feature scaling
par(mfrow=c(3,4))
for(i in colnames(d)[-12]){
  hist(d[,i],main = i)
}

#MOdeling
ti=createDataPartition(d$quality,p=0.8,list = F)
train=d[ti,]
test=d[-ti,]

#Decision tree
dt=rpart(quality~.,train,method = 'anova')
dt_p=round(predict(dt,test[,-12]))
accuracy(test$quality,dt_p)
#mae=0.53
#mape=9.544

#Random forest
rf=randomForest(quality~.,train,ntree=200)
rf_p=round(predict(rf,test[,-12]))
accuracy(test$quality,rf_p)
#mae=0.289
#mape=5.22

#gradient boosting
gb=gbm(quality~.,data = train,n.trees = 250)
gb_p=round(predict(gb,test[,-12],n.trees = 250))
accuracy(test$quality,gb_p)
#mae=0.40
#mape=7.27

#xgboost
xg=xgboost(data = as.matrix(train[,-12]),label = as.matrix(train[,12]),nrounds = 50,objective='reg:linear')
xg_p=round(predict(xg,as.matrix(test[,-12])))
accuracy(test$quality,xg_p)
#mae=0.36
#mape=6.62

#linear regresion
lr=lm(quality~.,data = train)
summary(lr)
lr_p=round(predict(lr,test[,-12]))
accuracy(test$quality,lr_p)
#mae=0.41
#mape=7.38

#KNN
km=knn(train[,-12],test[,-12],train[,12],k=5)
km=as.numeric(km)
accuracy(test$quality,km)
#mae=2.01
#mape=59.19

#support vector machine
sv=svm(quality~.,train)
sv_p=round(predict(sv,test[,-12]))
accuracy(test$quality,sv_p)
#mae=0.39
#mape=7.18
