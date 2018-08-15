library(caret)
library(ROCR) 
library(tidyverse)
library(Amelia)
library(mlbench)
library(ggplot2)
library(reshape2)
library(e1071)
library(car)        #<-- used to get Prestige dataset; and 'symbox' function
library(EnvStats)   #<-- used to get "boxcox" function
library(ggbiplot)
library(MASS)
library(devtools)
library(ggord)
library(mice)
library(VIM)
library(Amelia)
library(randomForest)
library(rpart.plot)
library(gbm)

userfunction <- function(true,pred,eps = 1e-15){

predoutput=as.numeric(pred>0.5) # Considering 0.5 as threshold to give 1(yes) as output.
# Creating dataframe predvals with true values,predicted probabilities,Predicted output.
predVals <-  data.frame(trueVal=true, predClass=predoutput, predProb=pred)
#1. Confusion Matrix

confM <- confusionMatrix(predoutput,true, positive="1" ) #
print(confM)# To print confusion matrix

#2. K-S chart  (Kolmogorov-Smirnov chart) 
# measures the degree of separation between the positive (y=1) and negative (y=0) distributions

predVals$group<-cut(predVals$predProb,seq(1,0,-.1),include.lowest=T)
xtab<-table(predVals$group,predVals$trueVal)
print(xtab)

#make empty dataframe
KS<-data.frame(Group=numeric(10),
               CumPct0=numeric(10),
               CumPct1=numeric(10),
               Dif=numeric(10))

#fill data frame with information: Group ID, 
#Cumulative % of 0's, of 1's and Difference
for (i in 1:10) {
  KS$Group[i]<-i
  KS$CumPct0[i] <- sum(xtab[1:i,1]) / sum(xtab[,1])
  KS$CumPct1[i] <- sum(xtab[1:i,2]) / sum(xtab[,2])
  KS$Dif[i]<-abs(KS$CumPct0[i]-KS$CumPct1[i])
}

print(KS)

KS[KS$Dif==max(KS$Dif),]

maxGroup<-KS[KS$Dif==max(KS$Dif),][1,1]

#and the K-S chart
print(ggplot(data=KS)+
  geom_line(aes(Group,CumPct0),color="blue")+
  geom_line(aes(Group,CumPct1),color="red")+
  geom_segment(x=maxGroup,xend=maxGroup,
               y=KS$CumPct0[maxGroup],yend=KS$CumPct1[maxGroup])+
  labs(title = "K-S Chart", x= "Deciles", y = "Cumulative Percent"))

#3. ROC Curve AUC

predic <- prediction(pred, true)    #ROC curve for training data
perf <- performance(predic,"tpr","fpr") 
plot(perf,colorize=TRUE, print.cutoffs.at = c(0.25,0.5,0.75)); 
abline(0, 1, col="red") 
pos <- pred[true == '1']
neg <- pred[true == '0']
#Find the probability
set.seed(123)
p <- replicate(100000, sample(pos, size=1) > sample(neg, size=1))
#AUC value
(AUC <-mean(p))
print("AUC VALUE")
print(AUC)
#4. Distribution of true positives & true negatives:
plot(0,0,type="n", xlim= c(0,1), ylim=c(0,7),     
     xlab="Prediction", ylab="Density",  
     main="How well do the predictions separate the true positives & true negatives")

for (runi in 1:length(predic@predictions)) {
  lines(density(predic@predictions[[runi]][predic@labels[[runi]]==1]), col= "blue")
  lines(density(predic@predictions[[runi]][predic@labels[[runi]]==0]), col="green")
}

# 6.D statistic

predvals.1<-predVals[predVals$trueVal==1,] # dataframe having observations where y=yes(1)
predvals.0<-predVals[predVals$trueVal==0,] # dataframe having observations where y=no(0)
DStat <- mean(predvals.1$predProb) - mean(predvals.0$predProb) # calculating D statistic
print("D Statistics")
print(DStat) # print D statistic


#7. cumulative gain chart
gain <- performance(prediction(pred, true),"tpr","rpp")
gain1 <- performance(prediction(true, true),"tpr","rpp")

plot(x=c(0, 1), y=c(0, 1), type="l", col="red", lwd=2,
     ylab="True Positive Rate", 
     xlab="Rate of Positive Predictions")

gain.x = unlist(slot(gain, 'x.values'))
gain.y = unlist(slot(gain, 'y.values'))

lines(x=gain.x, y=gain.y, col="orange", lwd=2)

gain1.x = unlist(slot(gain1, 'x.values'))
gain1.y = unlist(slot(gain1, 'y.values'))

lines(x=gain1.x, y=gain1.y, col="darkgreen", lwd=2)

#5. Concordance, Discordance, 
# Get all actual observations and their fitted values into a frame
fitted<-data.frame(cbind(true,pred))
colnames(fitted)<-c('true_values','pred_prob')
# Subset only ones
ones<-fitted[fitted[,1]==1,]
# Subset only zeros
zeros<-fitted[fitted[,1]==0,]

# Initialise all the values
pairs_tested<-nrow(ones)*nrow(zeros)
conc<-0
disc<-0

# Get the values in a for-loop
for(i in 1:nrow(ones))
{
  conc<-conc + sum(ones[i,"pred_prob"]>zeros[,"pred_prob"])
  disc<-disc + sum(ones[i,"pred_prob"]<zeros[,"pred_prob"])
}
# Calculate concordance, discordance and ties
concordance<-conc/pairs_tested
discordance<-disc/pairs_tested
ties_perc<-(1-concordance-discordance)

return(list("Concordance"=concordance,
            "Discordance"=discordance,
            "Tied"=ties_perc,
            "Pairs"=pairs_tested))

}

#8. Loglossbinary
LogLossBinary <- function(true, pred, eps = 1e-15) {
  predicte = pmin(pmax(pred, eps), 1-eps)
  - (sum(true * log(pred) + (1 - true) * log(1 - pred))) / length(true)
}
# END OF THE FUNCTION
#------------------------------------------------------------------------------------------------
#Q2.
#) Perform a basic EDA for the available data including visualizations, outlier analysis, 
#correlation analysis, PCA, etc., as you deem appropriate to understand the data.  (8 points) 

read_train <- read.csv('../../Assignment 5/LoanDefault-train.csv')
loan_train <- as_tibble(read_train)
read_test <- read.csv('../../Assignment 5/LoanDefault-Test.csv')
loan_test <- as_tibble(read_test)
colSums(is.na(loan_train))
colSums(is.na(loan_test))
#No Missing Values in Test and train

#Visualization
#box-plot for Education vs Limit
ggplot(data = loan_train, mapping = aes(x = Education, y = Limit)) +
  geom_boxplot()
#Histogram of Limit of Education Category with Defaulters Identification
ggplot(data = loan_train) +
  geom_histogram(mapping = aes(x = Limit/1000, fill = loan_train$Default), binwidth = 100) +
  facet_wrap(~Education)
#people with higher limit tend to default less than those with low limit. which can be clearly understood because banks tend to increase limit for those who make timely payments
boxplot(data=loan_train, loan_train$Limit ~ loan_train$Default,
                                          main = "Default vs Limit ",      
                                          xlab = "Default Status", 
                                          ylab = "Limit",
                                          col = "blue")
#Bill2 Vs Payment 1 : Bill Generated in this month needs to be paid in next month. 
#according to the ratio of the payers are more than the Defaulters
ggplot(loan_train, mapping = aes(x = Bill2, y = Payment1, color = Default)) +
  geom_point() 
#Density Plot
ggplot(loan_train, aes(x = Education, fill = Default)) + 
  geom_density() + 
  xlab("Default Payment Status") + ylab("Customer Count") 

#Correlation:
numerical_var = names(loan_train)[which(sapply(loan_train, is.numeric))]
loan_cor_numerics = cor(na.omit(loan_train[,numerical_var]))
col <- colorRampPalette(c("#BB4444","#EE9988","#FFFFFF","#77AADD","#4477AA"))
corrplot:: corrplot(loan_cor_numerics, method="circle", insig = "blank",shade.col=NA,tl.srt=45)

#Outlier Detection using histograms:
ggplot(loan_train) + 
  geom_boxplot(mapping = aes(x = Gender, y = Limit))
ggplot(loan_train) + 
  geom_boxplot(mapping = aes(x = Education, y = Bill1))


#PCA
pca_var <- c("Limit","Age","Bill1","Bill2","Bill3","Bill4","Bill5","Bill6","Payment1","Payment2","Payment3","Payment4","Payment5","Payment6")
loan_num_var <- dplyr :: select(loan_train, one_of(numerical_var))
pc <- prcomp(loan_num_var, scale = T)
summary(pc) # To get summary of principal components.
par(mfrow=c(1,1))
plot(pc) # plots pc
#First two columns itself  explains 51% of the variance in the whole matrix.
ggbiplot(pc, circle=T,obs.scale=1,varname.size=5, alpha = 0) + ylim(-7,7) + xlim(-7.5,6)
#status 1 to 6, Age, Limit more towards PC2 and 
#Bill 1 to 6 more towards PC1
#payment 1 to 6 participating in PC1 & PC2

#BASIC GLM
trainindex<-createDataPartition(loan_train$Default, p = 0.8, list=F)
train<-loan_train[trainindex,]
valid<-loan_train[-trainindex,]

fit <- glm(data=train, Default~., family="binomial")
summary(fit)
#) Choose three coefficients and describe their meaning (do they make intuitive sense) 
#Limit is clearly significant. Negative coefficient indicates as the Limit increases by one 
#probability of default payment for the next month decreases
#GenderM is a categorical variable and it is least significant
#Status 4,5,6 are not at all siginificant
#NULL DEVIANCE is deviance if you have an empty model
#RESIUDAL DEVIANCE: deviance based on actual model

#) Take a look at the residuals, influence measures, and variance inflation to see if you find anything 
#now let's take a look at the residuals

pearsonRes <-residuals(fit,type="pearson")
devianceRes <-residuals(fit,type="deviance")
rawRes <-residuals(fit,type="response")
studentDevRes<-rstudent(fit)
fv<-fitted(fit)

influence.measures(fit)
influencePlot(fit)
#From InfluencePlot we can see that 775, 3154 are the outliers
plot(predict(fit),residuals(fit))
vif(fit)
#Variance Inflation factor is a standard error which is very high for Bill1 to Bill6
#the "fit" glm object has a lot of useful information
names(fit)

head(fit$data)             # all of the data is stored
head(fit$y)                # the "true" value of the binary target  
head(fit$fitted.values)    # the predicted probabilities
fit$deviance               # the residual deviance

trueVa <- ifelse( valid$Default=="Y", 1, 0)
valid_t <- predict(fit, type = "response", newdata = valid)
userfunction(trueVa, valid_t)
LogLossBinary(trueVa, valid_t)
#-------------------------------------------------------------
#Q 3.

#FEATURE ENGINEERING
#Cumulative weighted sum of difference between bills and payments
train$BillDiffPay <- (0.6*(train$Bill1-train$Payment1) + 0.5*(train$Bill2-train$Payment2)+
                        0.4*(train$Bill3-train$Payment3)+0.3*(train$Bill4-train$Payment4)+
                        0.2*(train$Bill5-train$Payment5)+0.1*(train$Bill6-train$Payment6))

#sum of weighted status of all the previous 6 months.
train$status <-  0.9*train$Status1 + 0.7*train$Status2 + 0.45*train$Status3 + 0.4*train$Status4 +
  0.2*train$Status5 + 0.1*train$Status6

train$sampl  <-  (train$Limit)/(((train$Bill1-train$Payment1)*0.6*train$Status1)+
                                  ((train$Bill2-train$Payment2)*0.5*train$Status2)+
                                  ((train$Bill3-train$Payment3)*0.4*train$Status3)+
                                  ((train$Bill4-train$Payment4)*0.3*train$Status4)+
                                  ((train$Bill5-train$Payment5)*0.2*train$Status5)+
                                  ((train$Bill6-train$Payment6)*0.1*train$Status6))
#removing Payment 1 and Bill 6
exclude_train <- c('Payment1','Bill6')
excludevars <- names(loan_train) %in% exclude_train
train <- train[!excludevars]

### elastic net regression

# Tuning hyper parameters for the grid.
lambda.grid<-seq(0.0001,0.04,length=50)
alpha.grid<-seq(0,1,length=10)

srchGrd <-expand.grid(.alpha=alpha.grid, .lambda=lambda.grid)

ctrl <- trainControl(method = 'repeatedcv',number = 10,repeats = 5,
                     savePredictions = T, summaryFunction = twoClassSummary, classProbs = T)
#call caret function to perform CV using elastic net
elastic_model<-train(Default ~.,data=train, 
                     method="glmnet",
                     tuneGrid = srchGrd,
                     trControl=ctrl) #train is also a function in caret.

plot(elastic_model)  #look at CV error for different values of alpha and lambda

elastic_model$bestTune  #which was the best model

elastic_model_best <- elastic_model$finalModel   #save best model (alpha value)
coef(elastic_model_best, s = elastic_model$bestTune$lambda)  #look at coefficients associated with best lambda.

pred_el <- predict(elastic_model, newdata=valid,type = "prob") #To predict the probabilities of response variable
trueVa <- ifelse( valid$Default=="Y", 1, 0)
userfunction(trueVa, pred_el$Y)
LogLossBinary(trueVa, pred_el$Y)

#----------------------------------
#Random Forest  
set.seed(123)
# corelation between variables to reject unimportant variables


# Model for tuning creating customRF
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes
customRF
best_mtry <- tuneRF(train[,-25], train$Default,mtryStart = 10, 
                    ntreeTry=1600, 
                    stepFactor = 2, 
                    improve = 0.0001, 
                    doBest = TRUE,
                    nodesize = 30, 
                    importance=TRUE)

grid <- expand.grid(mtry = 10, ntree = 500 )
ctrl <- trainControl(method = "cv", number = 10, summaryFunction = twoClassSummary,classProbs = TRUE)

rf_fit <- train(Default ~ ., data = train,
                method = customRF,
                preProcess = c("center", "scale"),
                tuneGrid = grid,
                trControl = ctrl,         
                family= "binomial",
                metric= "ROC" #define which metric to optimize metric='RMSE'
)


rf_fit$results 
rf_fit$bestTune            # Best parameter value
rf_fit$finalModel         #Best Model

v_pred <- predict(rf_fit, valid,type = "prob")
trueVa <- ifelse( valid$Default=="Y", 1, 0)
userfunction(trueVa, v_pred$Y)
LogLossBinary(trueVa, v_pred$Y)
#########################################################################
#Decision Tree
####################
#Decision Tree rpart Model form Caret package and prepare resampling method
ctrl <-  trainControl(method = 'repeatedcv',number = 10,repeats = 5,
                      savePredictions = T, summaryFunction = twoClassSummary, classProbs = T)
set.seed(123)

#Information Gain
decision_model <- train(Default~., data=train, 
                        method="rpart", 
                        parms = list(split = "information"),  #information gain,
                        tuneLength = 10,
                        trControl=ctrl)
trueV <- ifelse( loan_train$Default=="Y", 1, 0)
prob_pp <- predict(decision_model,  loan_train, type="prob")
LogLossBinar(trueV, prob_pp$Y, eps = 1e-15 )

#Gini Index
decision_model_gini <- train(Default ~., data = loan_train, 
                             method = "rpart",
                             parms = list(split = "gini"),   #gini index
                             trControl=ctrl,
                             tuneLength = 10)

prob_pp_gini <- predict(decision_model_gini,  loan_train, type="prob")
LogLossBinar(trueV, prob_pp_gini$Y, eps = 1e-15 )


# display results
summary(decision_model_gini)
plot(decision_model_gini)
##########################################
#ADA BOOST

#creating grid for the parameters
grid <- expand.grid(mfinal = (1:6)*6, maxdepth = c(1, 6),
                    coeflearn = c("Breiman", "Freund", "Zhu"))
#cross validation
ctrl <-  trainControl(method = 'repeatedcv',number = 10,repeats = 5, classProbs = TRUE,
                      summaryFunction = twoClassSummary, savePredictions = T,returnResamp = "all")
#Ada boost model
adaboost_model <- train(Default~., data = train,
                        method = "AdaBoost.M1", 
                        trControl = ctrl,
                        tuneGrid = grid,
                        metric = "ROC", 
                        preProc = c("center", "scale")
)



Prediction.adaboost <- predict(adaboost_model, data = valid, type = 'prob')
userfunction(trueVal, Prediction.adaboost$Y)
LogLossBinary(trueVal, Prediction.adaboost$Y)

#Gradient Boosting Method


require(gbm)

read_train <- read.csv('../../Assignment 5/LoanDefault-train.csv')
trainD <- as_tibble(read_train)
read_test <- read.csv('../../Assignment 5/LoanDefault-Test.csv')
testD <- as_tibble(read_test)
#Feature Engineering for training data
trainD$billminuspay <-  ((trainD$Bill1+trainD$Bill2+trainD$Bill3+trainD$Bill4+trainD$Bill5+trainD$Bill6)-
                           (trainD$Payment1+trainD$Payment2+trainD$Payment3+trainD$Payment4+trainD$Payment5+trainD$Payment6))

trainD$status <-  (0.6*trainD$Status1 + 0.55*trainD$Status2 + 0.35*trainD$Status3 + 0.3*trainD$Status4 +
                     0.2*trainD$Status5 + 0.1*trainD$Status6)

trainD$totpay  <- (( trainD$Payment1*0.6+trainD$Payment2*0.55+trainD$Payment3*0.35+trainD$Payment4*0.3+trainD$Payment5*0.2+trainD$Payment6*0.1)/6)
trainD$totBill  <- ((trainD$Bill6*0.1+trainD$Bill5*0.2+trainD$Bill4*0.3+trainD$Bill3*0.35+trainD$Bill2*0.55+trainD$Bill1*0.6)/6)

#Feature Engineering for Test data
testD$billminuspay <-  ((testD$Bill1+testD$Bill2+testD$Bill3+testD$Bill4+testD$Bill5+testD$Bill6)-
                          (testD$Payment1+testD$Payment2+testD$Payment3+testD$Payment4+testD$Payment5+testD$Payment6))

testD$status <-  (0.6*testD$Status1 + 0.55*testD$Status2 + 0.35*testD$Status3 + 0.3*testD$Status4 +
                    0.2*testD$Status5 + 0.1*testD$Status6)

testD$totpay  <- (( testD$Payment1*0.6+testD$Payment2*0.55+testD$Payment3*0.35+testD$Payment4*0.3+testD$Payment5*0.2+testD$Payment6*0.1)/6)
testD$totBill  <- ((testD$Bill6*0.1+testD$Bill5*0.2+testD$Bill4*0.3+testD$Bill3*0.35+testD$Bill2*0.55+testD$Bill1*0.6)/6)



#converting y,n to 1,0
trainD$Default<-ifelse(trainD$Default=='Y', 1,0)

#visualization
plt1 <- ggplot(trainD, aes(x = trainD$Limit, y = trainD$billminuspay, colour = as.factor(trainD$Default))) +
  geom_point(size=3) +
  ggtitle("Wines")
plt1

#separating training and test data
train=sample(1:16000,size=10000)

#names(testD)

loan.boost=gbm(Default ~ . ,data = trainD[train,],distribution = "gaussian",n.trees = 1000,cv.folds = 5,
               shrinkage = 0.01, interaction.depth = 4,n.cores = 3)
loan.boost

summary(loan.boost) #Summary gives a table of Variable Importance and a plot of Variable Importance


summary(loan.boost)

#Plotting the Partial Dependence Plot
#The partial Dependence Plots will tell us the relationship and dependence of the variables \(X_i\) with the Response variable \(Y\).

#Plot of Response variable with lstat variable
plot(loan.boost,i="status") 
#Inverse relation with lstat variable

plot(loan.boost,i="Status1") 
#as the average number of rooms increases the the price increases

#Prediction on Test Set
#We will compute the Test Error as a function of number of Trees.
n.trees = seq(from=100 ,to=1000, by=25) #no of trees-a vector of 100 values 

#Generating a Prediction matrix for each Tree
predmatrix<-predict(loan.boost,trainD[-train,],n.trees = n.trees)
#Calculating The Mean squared Test Error
data_t = trainD[-train,]
true<- data_t$Default
test.error<-with(data = data_t,apply( (predmatrix-(true))^2,2,mean))
head(test.error) #contains the Mean squared test error for each of the 100 trees averaged

#Plotting the test error vs number of trees

plot(n.trees , test.error , pch=19,col="blue",xlab="Number of Trees",ylab="Test Error", main = "Perfomance of Boosting on Test Set")

#adding the RandomForests Minimum Error line trained on same data and similar parameters
abline(h = min(test.error),col="red") #test.err is the test error of a Random forest fitted on same data
legend("topright",c("Minimum Test error Line for Random Forests"),col="red",lty=1,lwd=1)
min(test.error)


#prediction on train to calculate log loss
predmatrix_logloss<-predict(loan.boost,trainD[-train,],n.trees = 800)
dim(predmatrix_logloss) #dimentions of the Prediction Matrix
#(predmatrix)
#predmatrix_logloss <- predmatrix_logloss-1

LogLossBinary(true, predmatrix_logloss, eps = 1e-15 )
userfunction(true, predmatrix_logloss)z
#Generating a Prediction matrix for each Tree for final test dataset
pred_1000<-predict(loan.boost,testD,n.trees = 800)
dim(pred_1000) #dimentions of the Prediction Matrix
(pred_1000)
#Final Solution
Id <- 1:14000
solutiongbmbest945am14th <- data.frame(Id,Default=pred_1000)
write.csv(solutiongbmbest945am14th,"gbm.csv")
