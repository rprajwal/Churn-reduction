rm(list = ls())

#loading libraries
library(stringr)
library(tm)
library(wordcloud)
library(ggplot2)
library(e1071)
library(C50)
library(caret)
library(textstem)
library(randomForest)
library(xgboost)

#importing data
d=read.csv('D:/analytics/practice/spam.csv')

#removing unwanted variables
d=d[,1:2]

#checking the data types of variables
str(d)

#Renaming the names of variables
colnames(d)=c('spam','Text')

#Converting spam variable to numerical type
d$spam=gsub('ham',0,d$spam)
d$spam=gsub('spam',1,d$spam)
d$spam=as.factor(d$spam)

#Checking number of texts in each class
ggplot(d,aes_string(d$spam))+geom_bar()

#Removing empty spaces from start and end of the string
d$Text=str_trim(d$Text)

#Converting texts into corpus
c=Corpus(VectorSource(d$Text))

#case folding
c=tm_map(c,tolower)

#remove punctuation marks
c=tm_map(c,removePunctuation)

#remove numbers
c=tm_map(c,removeNumbers)

#remove stopwords
c=tm_map(c,removeWords,stopwords('english'))

#remove blank spaces
c=tm_map(c,stripWhitespace)

#lemmatization
c=tm_map(c,lemmatize_strings)

#Converting corpus back to dataframe
d$ptext=get('content',c)

#plotting wordcloud to findout most frequent words in spam and ham texts
#words in spam messages
wordcloud(d[d$spam==1,'ptext'],random.order = F,colors = brewer.pal(8,'Dark2'))

#words in non spam messages
wordcloud(d[d$spam==0,'ptext'],random.order = F,colors = brewer.pal(12,'Paired'))

#There are some stopwords which are still remaining.
c=tm_map(c,removeWords,c('ill','will','now','just',stopwords('en')))
c=tm_map(c,stripWhitespace)

#build term document matrix
tdm=TermDocumentMatrix(c)

#removing sparse terms from term document matrix
rst=removeSparseTerms(tdm,0.999)

#Converting term document matrix into a data frame
tdd=as.data.frame(t(as.matrix(rst)))

#Finding most frequent words
freq=findFreqTerms(rst,lowfreq = 5)

#Reducing the size of data frame so that it should have terms which appears more than 10 times 
tdd=tdd[,freq]
tdd=cbind.data.frame(tdd,TT=d$spam)
names(tdd)=make.names(names(tdd))

#Splitting data to train test
ti=sample(nrow(d),size = 0.8*nrow(d))
train=tdd[ti,]
test=tdd[-ti,]


#Modeling
#Random Forest
rf=randomForest(TT~.,data = train,ntree=100)
rfp=predict(rf,test[,-1149])
confusionMatrix(test[,1149],rfp)

#Accuracy=97.94
#False negative rate=12.5
#False positive rate=0.4

#------------------------------------------------------

#Support vector machine
sv=svm(TT~.,train,cost=10)
svp=predict(sv,test[,-1149])
confusionMatrix(test[,1149],svp)

#Accuracy=96.32
#False negative rate=27.3
#False positive rate=0

#---------------------------------------------------

#Logistic regression
lr=glm(TT~.,data=train,family = 'binomial')
summary(lr)
lrp=predict(lr,test[,-1149])
lrp=ifelse(lrp<0.5,0,1)
table(test[,1149],lrp)

#Accuracy=93.9
#False negative rate=13.3
#False positive rate=4.9

#---------------------------------------------------------

#Naive Bayes
nb=naiveBayes(TT~.,train)
nbp=predict(nb,test[,-1149])
confusionMatrix(test$TT,nbp)

#Accuracy=13.63
#False negative rate=100
#False positive rate=0

#------------------------------------------------------

#xgboost
xg=xgboost(data = as.matrix(train[,-1149]),label = as.matrix(train[,1149]),objective='binary:logistic',nrounds = 100)
xgp=predict(xg,as.matrix(test[,-1149]))
xgp=ifelse(xgp>0.5,1,0)
table(test[,1149],xgp)

#Accuracy=97.84
#False negative rate=14.4
#False positive rate=0.2