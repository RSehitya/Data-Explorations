#####******************************************######

#kick off classification techniques

cat("\014") 
#required for Random Forest Classification Technique

library(randomForest)

#required for Support Vector Machine Classification Technique
library(e1071)

#required for conditional inference trees
library(party)

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

trumpcat<-read.csv("~/GitHub/Data-Explorations/2016-Presidential-Elections/Republican Primary Analysis/Classification Rules/Trump Categorical.csv")
str(trumpcat)

cruzcat<-read.csv("~/GitHub/Data-Explorations/2016-Presidential-Elections/Republican Primary Analysis/Classification Rules/Cruz Categorical.csv")
str(cruzcat)

kasichcat<-read.csv("~/GitHub/Data-Explorations/2016-Presidential-Elections/Republican Primary Analysis/Classification Rules/Kasich Categorical.csv")
str(kasichcat)

#set training and validation sets

set.seed(9000)
smp_size <- floor(0.80 * nrow(trumpcat))
train_ind <- sample(seq_len(nrow(trumpcat)), size = smp_size)
trumpcattrain <- trumpcat[train_ind,]
trumpcatvalid <- trumpcat[-train_ind,]

set.seed(9000)
smp_size <- floor(0.80 * nrow(cruzcat))
train_ind <- sample(seq_len(nrow(cruzcat)), size = smp_size)
cruzcattrain <- cruzcat[train_ind,]
cruzcatvalid <- cruzcat[-train_ind,]

set.seed(9000)
smp_size <- floor(0.80 * nrow(kasichcat))
train_ind <- sample(seq_len(nrow(kasichcat)), size = smp_size)
kasichcattrain <- kasichcat[train_ind,]
kasichcatvalid <- kasichcat[-train_ind,]

######Random Forest Analysis######################
set.seed(1234)
fit.forest<-randomForest(Winner~.,data=trumpcattrain,na.action = na.roughfix,importance=TRUE)
fit.forest
importance(fit.forest, type = 2)
forest.pred<-predict(fit.forest,trumpcatvalid)
forest.perf<-table(trumpcatvalid$Winner,forest.pred,dnn=c("Actual","Predicted"))
forest.perf
#85% accuracy with randomforest for trump
performance(forest.perf)

set.seed(1234)
cfit.forest<-randomForest(Winner~.,data=cruzcattrain,na.action = na.roughfix,importance=TRUE)
cfit.forest
importance(cfit.forest, type = 2)
cforest.pred<-predict(cfit.forest,cruzcatvalid)
cforest.perf<-table(cruzcatvalid$Winner,cforest.pred,dnn=c("Actual","Predicted"))
cforest.perf
#87% accuracy with randomForest for Ted Cruz
performance(cforest.perf)

set.seed(1234)
kfit.forest<-randomForest(Winner~.,data=kasichcattrain,na.action = na.roughfix,importance=TRUE)
kfit.forest
importance(kfit.forest, type = 2)
kforest.pred<-predict(kfit.forest,kasichcatvalid)
kforest.perf<-table(kasichcatvalid$Winner,kforest.pred,dnn=c("Actual","Predicted"))
kforest.perf
#98% accuracy with randomForest for Kasich
performance(kforest.perf)

######End Random Forest Analysis#################

###logistic regression####

tfit.logit<-glm(Winner~Age775214+Edu635213+Lfe305213+Lnd110210+Man450207+Nes010213+Pop010210+Pop645213+Pvy020213+Rhi225214+Rhi325214+Rhi825214+Rtn131207+Sbo015207+Sbo315207+Sbo415207+Sex255214,data=trumpcattrain, family=binomial())
summary(tfit.logit)
tprob<-predict(tfit.logit,trumpcatvalid,type="response")
tlogit.pred<-factor(tprob>0.5,levels=c(FALSE,TRUE),labels=c("Donald Trump","Not Donald Trump"))
tlogit.perf<-table(trumpcatvalid$Winner,tlogit.pred,dnn=c("Actual","Predicted"))
tlogit.perf
#76% accuracy
performance(tlogit.perf)

cfit.logit<-glm(Winner~Edu635213+Hsd310213+Lfe305213+Inc110213+Pop060210+Pop645213+Pop815213+Pvy020213+Sbo415207,data=cruzcattrain, family=binomial())
summary(cfit.logit)
cprob<-predict(cfit.logit,cruzcatvalid,type="response")
clogit.pred<-factor(tprob>0.5,levels=c(FALSE,TRUE),labels=c("Ted Cruz","Not Ted Cruz"))
clogit.perf<-table(cruzcatvalid$Winner,clogit.pred,dnn=c("Actual","Predicted"))
clogit.perf
#78% accuracy
performance(clogit.perf)

kfit.logit<-glm(Winner~Edu685213+Edu635213+Hsd310213+Sbo015207+Age775214+Pop715213+Pop060210+Pst120214,data=kasichcattrain, family=binomial())
summary(kfit.logit)
kprob<-predict(kfit.logit,kasichcatvalid,type="response")
klogit.pred<-factor(kprob>0.5,levels=c(FALSE,TRUE),labels=c("John Kasich","Not John Kasich"))
klogit.perf<-table(kasichcatvalid$Winner,klogit.pred,dnn=c("Actual","Predicted"))
klogit.perf
#96% accuracy
performance(klogit.perf)

#####End Logistic Regression#############


#####Support vector Machines###############

set.seed(1234)
tfit.svm<-svm(Winner~.,data=trumpcattrain)
tfit.svm
tsvm.pred<-predict(tfit.svm,na.omit(trumpcatvalid))
tsvm.perf<-table(na.omit(trumpcatvalid)$Winner,tsvm.pred,dnn=c("Actual","Predicted"))
tsvm.perf
#82% accuracy for trump with support vector machines
performance(tsvm.perf)

set.seed(1234)
cfit.svm<-svm(Winner~.,data=cruzcattrain)
cfit.svm
csvm.pred<-predict(cfit.svm,na.omit(cruzcatvalid))
csvm.perf<-table(na.omit(cruzcatvalid)$Winner,csvm.pred,dnn=c("Actual","Predicted"))
csvm.perf
#84% accuracy for cruz with support vector machines
performance(csvm.perf)

set.seed(1234)
kfit.svm<-svm(Winner~.,data=kasichcattrain)
kfit.svm
ksvm.pred<-predict(kfit.svm,na.omit(kasichcatvalid))
ksvm.perf<-table(na.omit(kasichcatvalid)$Winner,ksvm.pred,dnn=c("Actual","Predicted"))
ksvm.perf
#98% accuracy for Kasich with support vector machines
performance(ksvm.perf)

#####***End Support Vector Machine Analysis*****###########

####Conditional Inference Trees#####

tfit.ctree<-ctree(Winner~.,data=trumpcattrain)
plot(tfit.ctree,main="Trump Conditional Inference Tree")
tctree.pred<-predict(tfit.ctree,trumpcatvalid,type="response")
tctree.perf<-table(trumpcatvalid$Winner,tctree.pred,dnn=c("Actual","Predicted"))
tctree.perf
#76% accurace for trump
performance(tctree.perf)

cfit.ctree<-ctree(Winner~.,data=cruzcattrain)
plot(cfit.ctree,main="Cruz Conditional Inference Tree")
cctree.pred<-predict(cfit.ctree,cruzcatvalid,type="response")
cctree.perf<-table(cruzcatvalid$Winner,cctree.pred,dnn=c("Actual","Predicted"))
cctree.perf
#79% accurace for cruz
performance(cctree.perf)

kfit.ctree<-ctree(Winner~.,data=kasichcattrain)
plot(kfit.ctree,main="Kasich Conditional Inference Tree")
kctree.pred<-predict(kfit.ctree,kasichcatvalid,type="response")
kctree.perf<-table(kasichcatvalid$Winner,kctree.pred,dnn=c("Actual","Predicted"))
kctree.perf
#98% accurace for kasich
performance(kctree.perf)

####End Conditional Inference Trees#############