#loading laibraries
library(DMwR)
library(ggplot2)
library(dummies)
library(rpart)
library(forecast)
library(randomForest)
library(caret)
library(gbm)
library(e1071)

#importing data into dataframe
d=read.csv('D:/analytics/practice/BlackFriday.csv')

#Finding the data types of variables
#All the variables are categorical except target variable
str(d)

#Missing value analysis
mv=data.frame(apply(d, 2, function(x){sum(is.na(x))}))
colnames(mv)[1]='count'
mv$Percentage=(mv$count/nrow(d))*100

#imputing missing values with 0
d$Product_Category_2[is.na(d$Product_Category_2)]=0
d$Product_Category_3[is.na(d$Product_Category_3)]=0


#Exploratory data analysis

#Top 10 customers who have made more number of purchases
ed=aggregate.data.frame(d[,c('User_ID','Purchase')],by = list(d$User_ID),length)
uc=ed[order(-ed$User_ID),][1:10,1:2]
colnames(uc)[2]='count'
colnames(uc)[1]='User_ID'
barplot(uc$count,uc$User_ID,names.arg = uc$User_ID,xlab = 'User_ID',ylab = 'count')

#Top 10 highly sold products
tp=aggregate.data.frame(d,by=list(d$Product_ID),length)[,1:2]
colnames(tp)=c('Product_ID','count')
tp=tp[order(-tp$count),][1:10,]
barplot(tp$count,names.arg = tp$Product_ID,xlab = 'Product_ID',ylab = 'count')


#Finding out which gender has made highest purchase.
ggplot(d,aes_string(x='Gender'))+geom_bar()


#Finding out which age group has made highest purchase
ggplot(d,aes(x=d$Age))+geom_bar()

#Finding out people with what ocupation has made the highest purchase
ggplot(d,aes_string(x='Occupation'))+geom_bar()

#People belonging to city category has made maximum purchase
ggplot(d,aes_string(x=d$City_Category))+geom_bar()
+xlab('City_Category')

#People staying in the city from last 1 year have made highest number of purchases.
ggplot(d,aes_string(x='Stay_In_Current_City_Years'))+geom_bar()

#People with maital status 0 have made maximum purchases
ggplot(d,aes_string(x='Marital_Status'))+geom_bar()

#Products belonging to category 5 are the most sold products under Product_Category_1
ggplot(d,aes_string(x=d$Product_Category_1))+geom_bar()

#Most of the products under Product_Category_2 and 3 are unsold
ggplot(d,aes_string(x='Product_Category_2'))+geom_bar()

ggplot(d,aes_string(x='Product_Category_3'))+geom_bar()


#Data pre processing
#Removing User_ID and Product_ID from the data set
d=subset(d,select=-c(User_ID,Product_ID))
str(d)

#Converting all the characters in variables to numeric values
d$Gender=factor(d$Gender,labels = 1:2)
d$Age=factor(d$Age,labels = 1:7)
d$City_Category=factor(d$City_Category,labels = 1:3)
d$Stay_In_Current_City_Years=factor(d$Stay_In_Current_City_Years,labels = 1:length(unique(d$Stay_In_Current_City_Years)))

#Converting variables with integer type to factor type
for(i in colnames(d[,-10])){
  if (class(d[,i]) %in% c('integer','numeric')){
  print(i)
    d[,i]=as.factor(d[,i])
    }
}

#Splitting categorical variables
nd=dummy.data.frame(d,names=c("Age","Occupation","City_Category","Stay_In_Current_City_Years","Product_Category_1", 
                              "Product_Category_2","Product_Category_3"))

#-------------------------------------------------------

#Modeling
#Dividing the data as train and test
ti=sample(nrow(d),0.02*nrow(d))
train=nd[ti,]

tei=sample(nrow(d),0.01*nrow(d))
test=nd[tei,]

#--------------------------------------------------

#Decision tree
dt=rpart(Purchase~.,train,method = 'anova')
dt_p=predict(dt,test[,-91])
accuracy(dt_p,test[,91])
#mae=2399.088
#mape=41.066

#--------------------------------------------------

#Random forest
rf=randomForest(Purchase~.,train,ntree=100)
rf_p=predict(rf,test[,-91])
accuracy(rf_p,test[,91])
#mae=2282.92
#mape=34.36

#--------------------------------------------------

#Gradient boosting
gb=gbm(Purchase~.,data=train,n.trees = 300)
gb_p=predict(gb,test[,-91],n.trees = 300)
accuracy(gb_p,test[,91])
#mae=2365.95
#mape=37.94

#--------------------------------------------------

#Linear regression
lr=lm(Purchase~.,train)
lr_p=predict(lr,test[,-91])
accuracy(lr_p,test[,91])
#mae=2311.27
#mape=35.07

#--------------------------------------------------

#KNN
km=knn(train[,-91],test = test[,-91],train[,91],k = 5)
km_p=as.numeric(km)
accuracy(km_p,test[,91])
#mae=6617.06
#mape=68.59

#--------------------------------------------------

#sVM
sv=svm(Purchase~.,train)
sv_p=predict(sv,test[,-91])
accuracy(sv_p,test[,91])
#mae=2320.53
#mape=36.32

#--------------------------------------------------
#From above models it is clear that Random forest has the lowest mean absolute percentage error.
#Random forest performs better with this dataset.