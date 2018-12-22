#loading required libraries
library(corrplot)
library(C50)
library(caret)
library(randomForest)
library(class)
library(e1071)

#reading data from csv file
d=read.csv('D:/analytics/project/Train_data.csv')

#checking the shape of data
dim(d)

#structure of data
str(d)

#Converting string values to numeric values
d$state=factor(d$state,labels=1:length(levels(factor(d$state))))

d$international.plan=gsub(' no',0,d$international.plan)
d$international.plan=gsub(' yes',1,d$international.plan)

d$voice.mail.plan=gsub(' no',0,d$voice.mail.plan)
d$voice.mail.plan=gsub(' yes',1,d$voice.mail.plan)

d$Churn=ifelse(d$Churn==' False.',0,1)

#converting datatypes of few variables to proper datatypes
d$area.code=as.factor(d$area.code)
d$international.plan=as.factor(as.numeric(d$international.plan))
d$voice.mail.plan=as.factor(as.numeric(d$voice.mail.plan))
d$Churn=as.factor(d$Churn)

#verifying the structure of data
str(d)

#missing value analysis
mv=as.data.frame(apply(d,2,function(x){sum(is.na(x))}))
colnames(mv)[1]='count'

#No miising values found

#outlier analysis
par(mfrow=c(3,5))
for (i in colnames(d)) {
  if(class(d[,i]) %in% c('integer','numeric')){
    boxplot(d[,i],main=i)
  }
}
#since all the values look legit, will keep the data as is.

#continuous variables
con=c('account.length','number.vmail.messages','total.day.minutes',"total.day.calls","total.day.charge",
      "total.eve.minutes","total.eve.calls","total.eve.charge","total.night.minutes","total.night.calls","total.night.charge",
      "total.intl.minutes","total.intl.calls","total.intl.charge","number.customer.service.calls")

#categorical variables
categ=c("state","area.code","international.plan","voice.mail.plan")

#Feature selection
#correlation analysis
corrplot(cor(d[,con]),method = 'number')

#chi square test
for (i in categ) {
  print(i)
  print(chisq.test(table(d$Churn,d[,i])))
}
#p value of 'area code' is greater than 0.5, so our null hypothesis is true.
#That means area code is independent of target variable

d=subset(d,select = -c(total.day.minutes,total.eve.minutes,total.night.minutes,total.intl.minutes,area.code,phone.number))

#verifying data
dim(d)

#Feature scaling
#Normality check
par(mfrow=c(3,4))
for (i in colnames(d)) {
  if(class(d[,i]) %in% c('integer','numeric')){
    hist(d[,i],main=i)
  }
}
#since most of the variables are not uniformly distributed, data is not scaled.

#preparing train and test data
train=d

test=read.csv('D:/analytics/project/Test_data.csv')

test$state=factor(test$state,labels=1:length(levels(factor(test$state))))

test$international.plan=gsub(' no',0,test$international.plan)
test$international.plan=gsub(' yes',1,test$international.plan)

test$voice.mail.plan=gsub(' no',0,test$voice.mail.plan)
test$voice.mail.plan=gsub(' yes',1,test$voice.mail.plan)

test$Churn=ifelse(test$Churn==' False.',0,1)

test$area.code=as.factor(test$area.code)
test$international.plan=as.factor(as.numeric(test$international.plan))
test$voice.mail.plan=as.factor(as.numeric(test$voice.mail.plan))
test$Churn=as.factor(test$Churn)

test=subset(test,select = -c(total.day.minutes,total.eve.minutes,total.night.minutes,total.intl.minutes,area.code,phone.number))

#Model development
#Decision tree
dt=C5.0(Churn~.,train,trials=100)

#predicting values
p=predict(dt,test[,-15])

#confusion matrix
cm=table(test[,15],p)
tn=cm[1,1]
fp=cm[1,2]
fn=cm[2,1]
tp=cm[2,2]

confusionMatrix(cm)
FNR=fn/(fn+tp)
tpr=tp/(tp+fn)

#accuracy=95.62
#FNR=29.4
#Sensitivity=70.53

#-------------------------------------------------------

#Random forest
rf=randomForest(Churn~.,train,ntree=600)

#predicting values
p=predict(rf,test[,-15])

#confusion matrix
cm=table(test[,15],p)
tn=cm[1,1]
fp=cm[1,2]
fn=cm[2,1]
tp=cm[2,2]

confusionMatrix(cm)
FNR=fn/(fn+tp)
tpr=tp/(tp+fn)

#accuracy=94.99
#FNR=29.01
#Sensitivity=70.98

#------------------------------------------------------------

#Logistic regression
lr=glm(Churn~.,train,family = 'binomial')

#summary of model
summary(lr)

#predicting values
p=predict(lr,test[,-15])
p=ifelse(p>0.5,1,0)

#confusion matrix
cm=table(test[,15],p)
tn=cm[1,1]
fp=cm[1,2]
fn=cm[2,1]
tp=cm[2,2]

confusionMatrix(cm)
FNR=fn/(fn+tp)
tpr=tp/(tp+fn)

#accuracy=87.4
#FNR=84.37
#Sensitivity=15.62

#-------------------------------------------------------------

#KNN
km=knn(train[,-15],test[,-15],train[,15])

#confusion matrix
cm=table(test[,15],km)
tn=cm[1,1]
fp=cm[1,2]
fn=cm[2,1]
tp=cm[2,2]

confusionMatrix(cm)
FNR=fn/(fn+tp)
tpr=tp/(tp+fn)

#accuracy=80.26
#FNR=79.46
#Sensitivity=20.53

#-------------------------------------------------------------

#Naive bayes
nb=naiveBayes(Churn~.,train)

#predicting values
p=predict(nb,test[,-15])

#confusion matrix
cm=table(test[,15],p)
tn=cm[1,1]
fp=cm[1,2]
fn=cm[2,1]
tp=cm[2,2]

confusionMatrix(cm)
FNR=fn/(fn+tp)
tpr=tp/(tp+fn)

#accuracy=88.12
#FNR=71.4
#Sensitivity=28.57

#In this case even though Random forest has slightly lower accuracy than decision tree, it
#has low FNR and higher sensitivity when compared to decision tree.
#Hence will freeze Random forest model for this data set.