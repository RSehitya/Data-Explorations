#Load necessary libraries
#install.packages("randomForest")
#install.packages("car")
#install.packages("gvlma")
#install.packages("randomForest")
#install.packages("e1071")
#install.packages("party")
#install.packages("bootstrap")
rm(list=ls())

library(car)
library(MASS)
library(leaps)
library(gvlma)
library(boot)

#required for Random Forest Classification Technique

library(randomForest)

#required for Support Vector Machine Classification Technique
library(e1071)

#required for conditional inference trees
library(party)


masterdata<-read.csv("masterdata.csv")
str(masterdata)

#the function below cross validates a model's R square statistic by implementing k-fold cross-validation. The sample is divided into k subsamples. Each of the k subsamples serves as a hold
#out group, and the combined observations from the remaining k-1 subsamples serve as the training group.

crossvalidate <- function(fit,k=471){
  require (bootstrap)
  theta.fit <- function (x,y){lsfit (x,y)}
  theta.predict <- function (fit,x){cbind (1,x) %*%fit$coef}
  x <- fit$model[,2:ncol(fit$model)]
  y <- fit$model[,1]
  results <- crossval(x,y,theta.fit,theta.predict,ngroup = k)
  r2 <- cor(y,fit$fitted.values)^2
  r2cv <- cor(y,results$cv.fit)^2
  cat("Original R-Square =",r2,"\n")
  cat(k,"Fold Cross-Validated R-Square =",r2cv,"\n")
  cat("Change =",r2-r2cv,"\n")
}

#user defined function - assesses binary classification accuracy

performance<-function(table,n=2){
  if(!all(dim(table)==c(2,2)))
    stop("Must be a 2 x 2 table")
  tn= table[1,1]
  fp= table[1,2]
  fn= table[2,1]
  tp= table[2,2]
  sensitivity = tp/(tp+fn)
  specificity = tn/(tn+fp)
  ppp = tp/(tp+fn)
  npp = tn/(tn+fp)
  hitrate = (tp+tn)/(tp+tn+fp+fn)
  result<-paste("Sensitivity = ",round(sensitivity,n),
                "\nSpecificity = ",round(specificity,n),
                "\nPositive Predictive Value = ",round(ppp,n),
                "\nNegative Predictive Value = ",round(npp,n),
                "\nAccuracy = ",round(hitrate,n),"\n",sep="")
  cat(result)
}


EstimateCaliforniaPrimaryResults<-function(predictions,CApredictdata)
{
  i<-1
  sumcandidatevoterscount<-0
  while(i<=nrow(CApredictdata))
  {
    #as of 2012, 31% of eligible voting population is republican, let us assume a 50% turnout, 20% population is below 18
    adjustmentvalue<-0.31*0.5*0.2
    candidatevoterscount<-CApredictdata$Pop010210[i]*adjustmentvalue*predictions[i]
    i<-i+1
    sumcandidatevoterscount<-sumcandidatevoterscount+candidatevoterscount
  }
  results<-sumcandidatevoterscount/sum(CApredictdata$Pop010210*adjustmentvalue)
  return(results)
}

adjusttozero<-function(predictions){
  i<-1
  while(i<=nrow(CApredictdata))
  {
    if (predictions[i]<0) {predictions[i] = 0}
    else{predictions[i] = predictions[i]}
    i<-i+1
  }
  return(predictions)
}

#build linear multiple regression mode to predict voter share for trump
trump<-masterdata[masterdata$Candidate=="Donald Trump",5:54]
str(trump)
#run correlations
cor(trump)
#Drop Bza010213,Bza110213,Votes,rtn130207,vet605213,wtn220207,fips,hsg010214,hsg495213, bps030214,afn120207,
#from the outset to address multicollinearity

#build randomized training and validation sets for running regressions
set.seed(9000)
smp_size <- floor(0.75 * nrow(trump))
train_ind <- sample(seq_len(nrow(trump)), size = smp_size)
trumpLRtrain <- trump[train_ind,]
trumpLRvalid <- trump[-train_ind,]

#check whether everything looks good with the CSV data
str(trumpLRtrain)
head(trumpLRvalid)

#run base model
basefit<-lm(Fraction.Votes~.-Bza010213-Bza110213-Votes-Age295214-Rtn130207-Vet605213-Wtn220207-Fips-Hsg010214-Hsg495213-Bps030214-Afn120207,data=trumpLRtrain)
summary(basefit)

#the variables that are statistically significant for trump are: Income, Education, Age(over 65), women %, population density
#mean travel time between work, land area, manufacturing shipments 2007, nonemployer establishments, persons below poverty level,
#african american %, american indian, asian %, white %, retail sales per capita, total number of firms, women owned firms, hispanic owned firms

newfit<-lm(Fraction.Votes~Age775214+Edu635213+Edu685213+Lfe305213+Inc110213+Inc910213+Lnd110210+Man450207+Nes010213+Pop060210+Pvy020213+Rhi225214+Rhi325214+Rhi425214+Rhi825214+Rtn131207+Sbo015207+Sbo415207+Sbo515207+Sex255214,data=trumpLRtrain)
summary(newfit)

#look at akaike information criterion
stepAIC(newfit,direction="backward")

#look at regsubsets
bestfit<-regsubsets(Fraction.Votes~.,data=trumpLRtrain,nbest=8)
plot(bestfit,scale="adjr2")
#get rid of Nes010213 and Pop060210

bestfit2<-lm(Fraction.Votes~Age775214+Edu635213+Edu685213+Lfe305213+Inc110213+Inc910213+Lnd110210+Man450207+Pvy020213+Rhi225214+Rhi325214+Rhi425214+Rhi825214+Rtn131207+Sbo015207+Sbo415207+Sbo515207+Sex255214,data=trumpLRtrain)
summary(bestfit2)
#remove sbo515207

trumpbestfit3<-lm(Fraction.Votes~Age775214+Edu635213+Edu685213+Lfe305213+Inc910213+Lnd110210+Man450207+Pvy020213+Rhi225214+Rhi325214+Rhi425214+Rhi825214+Rtn131207+Sbo015207+Sbo415207+Sex255214,data=trumpLRtrain)
summary(trumpbestfit3)

trumpbestfit4<-lm(Fraction.Votes~Age775214+Edu635213+Edu685213+Lfe305213+Inc110213+Rhi225214+Rhi325214+Rhi425214+Rhi825214+Rtn131207+Sbo015207+Sbo415207+Sex255214,data=trumpLRtrain)
summary(trumpbestfit4)

#P value from the durbinwatson test, which is used to evaluate auto correlation & lack of error independence issues is not significant
#VERY good sign for bestfit3

durbinWatsonTest(trumpbestfit4)

par(mfrow=c(2,2))
plot(trumpbestfit4)
coefficients(trumpbestfit4)
crossvalidate(trumpbestfit4)

#time for predictions on validation set
predictions<-predict.lm(trumpbestfit4,newdata=trumpLRvalid)
rmse<-mean((trumpLRvalid$Fraction.Votes-predictions)^2)
print(sqrt(rmse))


#####***************************************************####

#build model for cruz
#build linear multiple regression mode to predict voter share for Kasich

cruz<-masterdata[masterdata$Candidate=="Ted Cruz",5:54]
str(cruz)
#run correlations
cor(cruz)
#Drop Bza010213,Bza110213,Votes,rtn130207,vet605213,wtn220207,fips,hsg010214,hsg495213, bps030214,afn120207,
#from the outset to address multicollinearity

#build randomized training and validation sets for running regressions
set.seed(9000)
smp_size <- floor(0.75 * nrow(cruz))
train_ind <- sample(seq_len(nrow(cruz)), size = smp_size)
cruzLRtrain <- cruz[train_ind,]
cruzLRvalid <- cruz[-train_ind,]

#check whether everything looks good with the CSV data
str(cruzLRtrain)
head(cruzLRvalid)

#run base model
basefit<-lm(Fraction.Votes~.-Bza010213-Bza110213-Votes-Age295214-Rtn130207-Vet605213-Wtn220207-Fips-Hsg010214-Hsg495213-Bps030214-Afn120207,data=cruzLRtrain)
summary(basefit)

#the variables that are statistically significant for cruz are: Income, Education, Age(over 65), population density
#mean travel time between work, foreign born population, language other than english spoken at home(spanish),person below poverty level
# hispanic owned firms

newfit<-lm(Fraction.Votes~Age775214+Edu635213+Hsd310213+Hsg445213+Lfe305213+Inc110213+Lnd110210+Man450207+Nes010213+Pop060210+Pop645213+Pop715213+Pop815213+Pvy020213+Rhi525214+Sbo415207+Sex255214,data=cruzLRtrain)
summary(newfit)

cruzfit<-lm(Fraction.Votes~Age775214+Edu635213+Hsd310213+Hsg445213+Lfe305213+Inc110213+Pop060210+Pop645213+Pop815213+Pvy020213+Sbo415207, data = cruzLRtrain)
summary(cruzfit)

#look at akaike information criterion
stepAIC(cruzfit,direction="backward")

#look at regsubsets
cruzbestfit<-regsubsets(Fraction.Votes~.,data=cruzLRtrain,nbest=2)
plot(cruzbestfit,scale="adjr2")

#P value from the durbinwatson test, which is used to evaluate auto correlation & lack of error independence issues is not significant
#VERY good sign for cruzfit

durbinWatsonTest(cruzfit)

par(mfrow=c(2,2))
plot(cruzfit)
coefficients(cruzfit)
crossvalidate(cruzfit)

#time for predictions on validation set for Ted Cruz
predictions<-predict.lm(cruzfit,newdata=cruzLRvalid)
rmse<-mean((cruzLRvalid$Fraction.Votes-predictions)^2)
print(sqrt(rmse))


####*********************************************************************#####

#build model for Kasich
#build linear multiple regression mode to predict voter share for Kasich

kasich<-masterdata[masterdata$Candidate=="John Kasich",5:54]
str(kasich)
#run correlations
cor(kasich)
#Drop Bza010213,Bza110213,Votes,rtn130207,vet605213,wtn220207,fips,hsg010214,hsg495213, bps030214,afn120207,
#from the outset to address multicollinearity

#build randomized training and validation sets for running regressions
set.seed(9000)
smp_size <- floor(0.75 * nrow(kasich))
train_ind <- sample(seq_len(nrow(kasich)), size = smp_size)
kasichLRtrain <- kasich[train_ind,]
kasichLRvalid <- kasich[-train_ind,]

#check whether everything looks good with the CSV data
str(kasichLRtrain)
head(kasichLRvalid)

#run base model
basefit<-lm(Fraction.Votes~.-Bza010213-Bza110213-Votes-Age295214-Rtn130207-Vet605213-Wtn220207-Fips-Hsg010214-Hsg495213-Bps030214-Afn120207,data=kasichLRtrain)
summary(basefit)

#the variables that are statistically significant for cruz are: Income, Education, Age(over 65), population density
#living in same house for over 1 year, language other than english spoken at home(spanish),population % change, person below poverty level, native hawai pacific islander firms owned
# hispanic owned firms

newfit<-lm(Fraction.Votes~Age775214+Edu635213+Hsd310213+Hsd410213+Inc110213+Inc910213+Pop010210+Pop060210+Pop715213+Pop815213+Pst120214+Pvy020213+Rhi525214+Sbo015207,data=kasichLRtrain)
summary(newfit)

newfit<-lm(Fraction.Votes~Age775214+Edu635213+Pop060210+Rhi525214+Sbo015207,data=kasichLRtrain)
summary(newfit)
#remove pvy020213(poverty indicator not significant for kasich), pop815213
kasichfit<-lm(Fraction.Votes~Edu685213+Edu635213+Hsd310213+Sbo015207+Age775214+Pop715213+Pop060210+Pst120214,data=kasichLRtrain)
summary(kasichfit)




#look at akaike information criterion
stepAIC(kasichfit,direction="backward")

#look at regsubsets
kasichbestfit<-regsubsets(Fraction.Votes~.,data=kasichLRtrain,nbest=2)
plot(kasichbestfit,scale="adjr2")

#P value from the durbinwatson test, which is used to evaluate auto correlation & lack of error independence issues is not significant
#VERY good sign for cruzfit

durbinWatsonTest(kasichfit)

par(mfrow=c(2,2))
plot(kasichfit)
coefficients(kasichfit)
crossvalidate(kasichfit)

#time for predictions on validation set for John Kasich
predictions<-predict.lm(kasichfit,newdata=kasichLRvalid)
rmse<-mean((kasichLRvalid$Fraction.Votes-predictions)^2)
print(sqrt(rmse))

#####****************************************************************#######
#predict california primary results based on our models

CApredictdata<-read.csv("California Primary County Stats.csv")
str(CApredictdata)
predictions<-predict.lm(trumpbestfit3,newdata=CApredictdata)

#trump

trumpvotingfraction<-EstimateCaliforniaPrimaryResults(predictions,CApredictdata)
#cruz

predictions<-predict.lm(cruzfit,newdata=CApredictdata)
predictions<-adjusttozero(predictions)
cruzvotingfraction<-EstimateCaliforniaPrimaryResults(predictions,CApredictdata)
#kasich

predictions<-predict.lm(kasichfit,newdata=CApredictdata)
predictions<-adjusttozero(predictions)
kasichvotingfraction<-EstimateCaliforniaPrimaryResults(predictions,CApredictdata)

#adjust for the fact that the models were based on data with 12 candidates, now there are only 3 candidates; normalize to 100

remainder<-1.00-trumpvotingfraction-cruzvotingfraction-kasichvotingfraction
total<-trumpvotingfraction+cruzvotingfraction+kasichvotingfraction

#it is likely that the votes of most of the other candidates who dropped out would go to cruz and kasich rather than trump, hence proportional allocation
#based on current performance may not be the best way to go here; best that can be done here is allocate equally.

trumpvotingfraction<-trumpvotingfraction+0.33*remainder
cruzvotingfraction<-cruzvotingfraction+0.33*remainder
kasichvotingfraction<-kasichvotingfraction+0.33*remainder
print(trumpvotingfraction)
print(cruzvotingfraction)
print(kasichvotingfraction)


#based on our regression models - trump will receive 48%, Cruz will get 36% and Kasich will secure 16% of the california primary vote
#hence, the prediction is that trump will win the california republican primary (Which he did :) Yay!!! The model predicts accurately)