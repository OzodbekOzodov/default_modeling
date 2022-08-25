rm(list=ls())
install.packages("pROC")
install.packages('caret')
install.packages('tidyverse')
install.packages('caret')
install.packages('InformationValue')
install.packages('ISLR')
install.packages("heuristica")
install.packages("mlbench")
install.packages('randomForest')
install.packages("truncnorm")
install.packages('fitdistrplus')
install.packages('rpart')
install.packages('tree')
install.packages('reprtree')
install.packages('randomForestExplainer')
install.packages('reprtree')


library(heuristica)
library(tidyverse)
library(caret)
library(InformationValue)
library(ISLR)
library(fitdistrplus)
library(rpart)

getwd()
#setwd("")  set working directory here
data = read.csv("UCI_Credit_card.csv")

#correlation table for necessary variables
cor_table= read.csv("Ozodbek_correlation_table.csv")
cor_table

#plot(density(data$AGE))

#sum(is.na(data))

data$male <- ifelse(data$SEX==1,1,0)
data$female <- ifelse(data$SEX==2,1,0)
data$married <- ifelse(data$MARRIAGE==1,1,0)
data$university_ed <- ifelse(data$EDUCATION==2,1,0)
data$duly <- ifelse(data$PAY_6==-1,1,0)
data$bills <- data$BILL_AMT1+data$BILL_AMT2+data$BILL_AMT3+data$BILL_AMT4+
  data$BILL_AMT5+data$BILL_AMT6

data$payment_status <- data$PAY_0+data$PAY_2+data$PAY_3+data$PAY_4+
  data$PAY_5+data$PAY_6

#data$default.payment.next.month <- (data$default.payment.next.month)

#multiple_reg <- lm(data$default.payment.next.month~
#                     data$married+data$PAY_6+data$PAY_AMT6) 

#summary(multiple_reg)


set.seed(123)

#training <- sample(
x <- 1:30000
x
training_index <- sample(x, size=15000, replace=FALSE)
training_index
training_set <- data[training_index,]
training_set
test_set <- data[-training_index,]
log_reg <- glm(training_set$default.payment.next.month~training_set$LIMIT_BAL+
                 training_set$payment_status+
                 training_set$bills+
                 training_set$married+
                 training_set$male+
                 training_set$duly+
                 training_set$AGE)
summary(log_reg)

predicted <- predict(log_reg,test_set)
predicted
#######################

## this is necessary to run simulation
log_reg_sim <- glm(training_set$default.payment.next.month~scale(training_set$LIMIT_BAL)+
                 scale(training_set$payment_status)+
                 scale(training_set$bills))
summary(log_reg_sim)
#######################

#find optimal cutoff probability to use to maximize accuracy

optimal <- optimalCutoff(training_set$default.payment.next.month, 
                         predicted)
optimal

training_set$defaults <- ifelse(predict(log_reg)>optimal,1,0)

log_cm <- caret:: confusionMatrix(as.factor(training_set$default.payment.next.month), 
                    as.factor(training_set$defaults))
log_cm


set.seed(123)
x <- 1:30000
### simple splitting into training and test data 
training2_index <- sample(x, 15000, replace=FALSE)  
training2_set <- data[training2_index,]  
test2_set <- data[-training2_index,]  
dim(test2_set)

training2_set$LIMIT_BAL <- scale(training2_set$LIMIT_BAL)
training2_set$bills <- scale(training2_set$bills)

# Cross validated training and test data
cv_train_log <- train(as.factor(default.payment.next.month) ~ bills + 
                         payment_status + LIMIT_BAL , data=data,
                      method='glm',
                      trControl=trainControl(method='cv', number=10, 
                                      savePredictions = TRUE ), tuneLength=30)
#+ married + male + AGE + duly                        
predicted_cv <- predict(cv_train_log, test2_set)

cm_cv <- caret::confusionMatrix(as.factor(test2_set$default.payment.next.month),
                               predicted_cv)

cm_cv

summary(cv_train_log)


plotdist(data$AGE)
varImp(cv_train_log)


#write.csv(data, "UCI_new.csv", row.names = TRUE)

plot(test2_set$bills, test2_set$predicted2)

###

# KNN
library(mlbench)
library(caret)
set.seed(123)
ind <- sample(x, 15000, replace=FALSE)  
ind
train_knn <- data[ind, ]
test_knn <- data[-ind,]
#knn model
trControl <- trainControl(method="repeatedcv",
                          number=10,
                          repeats = 3)
set.seed(123)
fit_knn <- train(default.payment.next.month~ bills+ 
               LIMIT_BAL + 
               payment_status,
             data=train_knn,
             method="knn",
             tuneLength=40,
             trControl=trControl,
             preProc=c('center', 'scale'))

summary(fit_knn)
fit_knn
plot(fit_knn)
varImp(fit_knn)

# model performance with test data
test_knn$defaults <- predict(fit_knn, newdata=test_knn)
test_knn$defaults
optimal_knn <- optimalCutoff(test_knn$default.payment.next.month, test_knn$defaults)

test_knn$defaults2 <- ifelse(test_knn$defaults>=optimal_knn, 1,0)
cm3_knn <- caret::confusionMatrix(as.factor(test_knn$default.payment.next.month) ,
                             as.factor(test_knn$defaults2))

cm3_knn

#model performance with all observations
data$knn_defaults <- predict(fit_knn, newdata=data)
data$knn_defaults
optimalCutoff(data$default.payment.next.month, data$knn_defaults)
data$knn_defaults_bin <- ifelse(data$knn_defaults>=0.48,1,0)
data$knn_defaults_bin
cm4_knn <- caret::confusionMatrix(as.factor(data$default.payment.next.month),
                             as.factor(data$knn_defaults_bin))

cm4_knn


#selecting more variables - KNN case 2
fit2 <- train(as.factor(default.payment.next.month)~ bills + LIMIT_BAL + payment_status
              + PAY_AMT6 + BILL_AMT5+university_ed+married,
              data=train_knn,
              method="knn",
              tuneLength=40,
              trControl=trControl,
              preProc=c('center', 'scale'))



summary(fit2)
fit2
plot(fit2)
varImp(fit2)

#model performance with test set
test_knn$defaults_fit2 <- predict(fit2, newdata=test_knn)
test_knn$defaults_fit2
optimalCutoff(test_knn$default.payment.next.month, test_knn$defaults_fit2)
test_knn$defaults_fit2_2 <- ifelse(test_knn$defaults_fit2>=0.47,1,0)
cm5_knn <- caret::confusionMatrix(as.factor(test_knn$default.payment.next.month) ,
                             as.factor(test_knn$defaults_fit2))

cm5_knn

#model performance with all observations
data$knn_defaults_fit2_all <- predict(fit2, newdata=data)
optimalCutoff(data$default.payment.next.month, data$knn_defaults_fit2_all)
data$knn_defaults_fit2_bin <- ifelse(data$knn_defaults_fit2_all>=0.47,1,0)
cm6_knn <- caret::confusionMatrix(as.factor(data$default.payment.next.month),
                             as.factor(data$knn_defaults_fit2_all))

cm6_knn


#Random Forests
# I refer to Random forest objects by adding rf

library(randomForest)
library(tidyverse)
library(tree)
library(reprtree)
library(randomForestExplainer)

set.seed(123)

ind_rf <- sample(x, 15000, replace=FALSE)  
train_rf <- data[ind_rf,]
test_rf <- data[-ind_rf,]

# convert integer variable default.payment.next.month into factor 
# variable so that random forest runs classification tree, not the
# regression one

train_rf$default.payment.next.month <- as.factor(train_rf$default.payment.next.month)
test_rf$default.payment.next.month <- as.factor(test_rf$default.payment.next.month)
data$default.payment.next.month <- as.factor(data$default.payment.next.month)

#model
fit_rf <- randomForest(default.payment.next.month ~ 
                         bills + 
                         LIMIT_BAL +
                         payment_status+ 
                         PAY_AMT6+ 
                         BILL_AMT6+
                         university_ed+
                         married, data=train_rf)


print(fit_rf)
varImp(fit_rf)
plot(fit_rf)


#the following two plots take a lot of time to run
# you can just find both on pdf file

plot_rf <- plot_min_depth_distribution(fit_rf)
plot_rf2 <- plot_importance_rankings(fit_rf)
plot_rf3 <- plot_min_depth_interactions(fit_rf)
plot_rf3


# test model
test_rf$rf_default <- predict(fit_rf, newdata = test_rf)
cm_rf <- caret:: confusionMatrix(as.factor(test_rf$default.payment.next.month),
                                      as.factor(test_rf$rf_default))
cm_rf

# using all observations

data$default_rf <- predict(fit_rf, newdata=data)
con_mat_rf2 <- caret::confusionMatrix(as.factor(data$default.payment.next.month),
                                      as.factor(data$default_rf))
con_mat_rf2


#___________________________________________________________________

#Simulation study
#data generating process

#keep necessary variables for further processing
emp_data <- data[, c(2,6,12,18,28,29,31,32)]
head(emp_data)

#create correlation table for covariance matrix
#in case any problems, use Ozodbek_correlation_table.csv 

cor_table <- cor(emp_data)
cor_table

#write.table(cor_table, file="correlation_table.csv")

# we need 8x8 vcov matrix to build multivariate normal distribution
dim(cor_table)


set.seed(123)
# set means
# all variables will be scaled and centered, therefore set all to zero
means_mu=c(0,0,0,0,0,0,0,0)
#create simulated data
data_sim <- mvrnorm(30000, mu = means_mu, Sigma = cor_table)

#assign names for clarification
colnames(data_sim)=c('LIMIT_BAL',
           'AGE',
           'PAY_6',
           'BILL_AMT6',
           'married',
           'university_ed',
           'bills',
           'payment_status')

data_sim <- data.frame(data_sim)

# the following keeps the marriage status and university education 
# dummies similar to the actual

data_sim$married <- ifelse(data_sim$married>0, 1, 0)
data_sim$university_ed <- ifelse(data_sim$university_ed>0, 1, 0)
head(data_sim)
cor(data_sim)


## create response variables type I and type II

set.seed(123)
errors <- runif(30000, -0.1, 0.1)
errors
gen <- predict(log_reg, data=data_sim)+errors
gen
data_sim$default_sim <- ifelse(gen>=0.3, 1, 0)
head(data_sim)
hist(data_sim$default_sim)
sum(data_sim$default_sim)

# Logistic regression 
set.seed(123)
x <- 1:30000
training_sim_inde <- matrix(NaN)
training_sim <- matrix(NaN)
test_sim <- matrix(NaN)
log_sim <- matrix(NaN)
p_values <- matrix(NaN)
default_sim2 <- matrix(NaN)
optimal_sim <- matrix(NaN)
default_sim_bin <- matrix(NaN)
errors <- matrix(NaN)
gen <- matrix(NaN)

###########
library(truncnorm)
##### Default values random allocation
bills <- matrix(NaN)
LIMIT_BAL <- matrix(NaN)
payment_status <- matrix(NaN) 
PAY_AMT6 <- matrix(NaN)
BILL_AMT5 <- matrix(NaN)
university_ed <- matrix(NaN)
married <- matrix(NaN)
male <- matrix(NaN)
female <- matrix(NaN)
defaults <- matrix(NaN)
log_model <- NaN
p_values <- matrix(NaN, nrow=10, ncol=4)
p_lim_b <- c(NaN)
p_payment <- c(NaN)
p_bills <- c(NaN)
p_pay <- c(NaN)
p_bill5 <- c(NaN)
predicted_sim <-matrix(NaN) 
predicted_bin <- matrix(NaN)
opt1 <- matrix(NaN)
predicted1 <- matrix(NaN)
default_sim_1 <- matrix(NaN)
default_sim2_1 <- matrix(NaN)
conf1_11 <- matrix(NaN)
conf1_12 <- matrix(NaN)
conf1_21 <- matrix(NaN)
conf1_22 <- matrix(NaN)

for (i in 1:1000){
  bills <- rweibull(30000, shape=1, scale=1)
  LIMIT_BAL <- rchisq(30000, 10)
  payment_status <- rtruncnorm(30000, a=-15, b=30, sd=5.8) 
  PAY_AMT6 <- rchisq(30000, 10)
  BILL_AMT5 <- rgamma(30000, shape=5)
  university_ed <- floor(rtruncnorm(30000, 0,2))
  married <- floor(rtruncnorm(30000, 0, 2))
  male <- floor(rtruncnorm(30000, 0,2))
  female <- ifelse(male==1,0,1)
  defaults <- floor(runif(30000,0,2))
  
  log_model <- glm(defaults~LIMIT_BAL+payment_status+bills+PAY_AMT6+BILL_AMT5)
  
  p_lim_b[i] <- summary(log_model)$coefficients[2,4]
  p_payment[i] <- summary(log_model)$coefficients[3,4]
  p_bills[i] <- summary(log_model)$coefficients[4,4]
  p_pay[i] <- summary(log_model)$coefficients[5,4]
  p_bill5[i] <- summary(log_model)$coefficients[6,4]
  
}
p_values <- cbind(p_lim_b, p_payment, p_bills)


length(which(p_lim_b<0.10))
length(which(p_payment<0.10))
length(which(p_bills<0.10))



length(which(p_lim_b<0.10 & p_payment<0.10))
length(which(p_lim_b<0.10& p_bills<0.10))

hist(p_lim_b)
hist(p_payment)
hist(p_bills)


######### multivariate normal distribution case
means_mu=c(0,0,0,0,0,0,0,0)
default_sim_2 <- matrix(NaN)
p_values <- matrix(NaN, nrow=10, ncol=4)
p_lim_b <- c(NaN)
p_payment <- c(NaN)
p_bills <- c(NaN)
p_pay <- c(NaN)
p_bill5 <- c(NaN)
LIMIT_BAL <- matrix(NaN)
predicted_sim <-matrix(NaN) 
predicted_bin <- matrix(NaN)
opt <- matrix(NaN)
predicted <- matrix(NaN)
default_sim <- matrix(NaN)
default_sim2 <- matrix(NaN)
conf_11 <- matrix(NaN)
conf_12 <- matrix(NaN)
conf_21 <- matrix(NaN)
conf_22 <- matrix(NaN)
#### this is necessary for predict function in the loop
log_reg_sim <- glm(training_set$default.payment.next.month~scale(training_set$LIMIT_BAL)+
                     scale(training_set$payment_status)+
                     scale(training_set$bills))
###______________________________________________________________

set.seed(123)
for (i in 1:1000){
  data_sim <- mvrnorm(30000, mu = means_mu, Sigma = cor_table)
  train_ind <- sample(1:30000, 15000, replace=FALSE)
  train_sim2 <- data_sim[train_ind,]
  test_sim2 <- data_sim[-train_ind,]
  def_log_train <- predict(log_reg_sim, data=train_sim2)+runif(15000,-0.15, 0.15)
  def_log_test <- predict(log_reg_sim, data=test_sim2)+runif(15000, -0.15,0.15)
  default_train <- ifelse(def_log_train>0.1855, 1, 0)
  default_test <- ifelse(def_log_test>0.1855, 1, 0)
  log_model2 <- glm(default_train~train_sim2[,1]+train_sim2[,7]+train_sim2[,8])
  p_lim_b[i] <- summary(log_model2)$coefficients[2,4]
  p_bills[i] <- summary(log_model2)$coefficients[3,4]
  p_pay[i] <- summary(log_model2)$coefficients[4,4]
  predicted <- predict(log_model2, data=test_sim2)
  opt <- optimalCutoff(default_train, predicted)
  predicted_bin <- ifelse(predicted>opt, 1, 0)
  conf_11[i] <- confusionMatrix(default_test, predicted_bin)[1,1]
  conf_12[i] <- confusionMatrix(default_test, predicted_bin)[1,2]
  conf_21[i] <- confusionMatrix(default_test, predicted_bin)[2,1]
  conf_22[i] <- confusionMatrix(default_test, predicted_bin)[2,2]
}


plot(density(def_log_train))
caret::confusionMatrix(as.factor(default_test), as.factor(predicted_bin))

conf_sim <- cbind(conf_11, conf_12, conf_21, conf_22)
colnames(conf_sim) <- c('cor nondef', 'wrong def', 'wrong nondef','corr def')
conf_sim <- data.frame(conf_sim)
conf_sim$accuracy <- (conf_11+conf_22)/(conf_11+conf_12+conf_21+conf_22)

hist(conf_sim$accuracy, main='Histogram of accuracy', xlab='Accuracy')

round(colMeans(na.omit(conf_sim)), 2)
max(conf_sim$accuracy)
which((conf_sim$accuracy)>0.57)
conf_sim


### KNN prediction with simulation results

trControl <- trainControl(method="repeatedcv",
                          number=10,
                          repeats = 3)
means_mu=c(0,0,0,0,0,0,0,0)
default_sim_2 <- matrix(NaN)
p_values <- matrix(NaN, nrow=10, ncol=4)
p_lim_b <- c(NaN)
p_payment <- c(NaN)
p_bills <- c(NaN)
p_pay <- c(NaN)
p_bill5 <- c(NaN)
LIMIT_BAL <- matrix(NaN)
predicted_sim <-matrix(NaN) 
predicted_bin <- matrix(NaN)
opt <- matrix(NaN)
predicted <- matrix(NaN)
default_sim <- matrix(NaN)
default_sim2 <- matrix(NaN)
data_sim2 <- matrix(NaN)
a_11 <- c(NaN)
a_12 <- c(NaN)
a_21 <- c(NaN)
a_22 <- c(NaN)
#### this is necessary for predict function in the loop
log_reg_sim <- glm(training_set$default.payment.next.month~scale(training_set$LIMIT_BAL)+
                     scale(training_set$payment_status)+
                     scale(training_set$bills))
###______________________________________________________________




for (i in 1:100){
  data_sim2 <- mvrnorm(30000, mu = means_mu, Sigma = cor_table)
  train_ind3 <- sample(1:30000, 15000, replace=FALSE)
  train_sim3 <- data_sim2[train_ind3,]
  test_sim3 <- data_sim2[-train_ind3,]
  def_knn_train <- predict(log_reg_sim, data=train_sim3)+runif(15000,-0.25, 0.20)
  def_knn_test <- predict(log_reg_sim, data=test_sim3)+runif(15000, -0.25, 0.20)
  default_train <- ifelse(def_knn_train>0.05, 1, 0)
  default_test <- ifelse(def_knn_test>0.05, 1, 0)
  train_knn=data.frame(train_sim3)
  train_knn$default <- default_train
  test_knn=data.frame(test_sim3)
  test_knn$default <- default_test
  
  knn_model_sim <- knn(train=train_knn, test=test_knn, cl=default_train, k=40)
  
  a <- confusionMatrix(knn_model_sim, test_knn$default)  
  a_11[i] <- a[1,1]
  a_12[i] <- a[1,2]
  a_21[i] <- a[2,1]
  a_22[i] <- a[2,2]
  
}
caret::confusionMatrix(as.factor(knn_model_sim), as.factor(test_knn$default))

confmat <- cbind(a_11, a_12, a_21, a_22)
confmat <- data.frame(confmat)
confmat$accuracy <- (a_11+a_22)/(a_11+a_12+a_21+a_22)
confmat
round(colMeans(confmat),2)

hist(confmat$accuracy, main="Histogram of accuracy in KNN", xlab="Accuracy")

caret::confusionMatrix(as.factor(knn_model_sim), as.factor(test_knn$default))




################ random forest simulation ###################

means_mu=c(0,0,0,0,0,0,0,0)
default_sim_2 <- matrix(NaN)
p_values <- matrix(NaN, nrow=10, ncol=4)
p_lim_b <- c(NaN)
p_payment <- c(NaN)
p_bills <- c(NaN)
p_pay <- c(NaN)
p_bill5 <- c(NaN)
LIMIT_BAL <- matrix(NaN)
predicted_sim <-matrix(NaN) 
predicted_bin <- matrix(NaN)
opt <- matrix(NaN)
predicted <- matrix(NaN)
default_sim <- matrix(NaN)
default_sim2 <- matrix(NaN)
data_sim2 <- matrix(NaN)
cm_11 <- c(NaN)
cm_12 <- c(NaN)
cm_21 <- c(NaN)
cm_22 <- c(NaN)

#### this is necessary for predict function in the loop
log_reg_sim <- glm(training_set$default.payment.next.month~scale(training_set$LIMIT_BAL)+
                     scale(training_set$payment_status)+
                     scale(training_set$bills))
###______________________________________________________________



library(randomForest)
set.seed(123)
for (i in 1:50){
  data_sim4 <- mvrnorm(30000, mu = means_mu, Sigma = cor_table)
  train_ind4 <- sample(1:30000, 15000, replace=FALSE)
  train_sim4 <- data_sim4[train_ind4,]
  test_sim4 <- data_sim4[-train_ind4,]
  def_rf_train <- predict(log_reg_sim, data=train_sim4)+runif(15000,-0.15, 0.15)
  def_rf_test <- predict(log_reg_sim, data=test_sim4)+runif(15000, -0.15, 0.15)
  default_train <- ifelse(def_rf_train>0.12, 1, 0)
  default_test <- ifelse(def_rf_test>0.12, 1, 0)
  train_rf=data.frame(train_sim4)
  train_rf$default <- default_train
  test_rf=data.frame(test_sim4)
  test_rf$default <- default_test
  
  model_rf <- randomForest(as.factor(train_rf$default)~train_rf$bills+
                             train_rf$payment_status+
                             train_rf$LIMIT_BAL)
  
  predicted_rf <- predict(model_rf, test_rf)
  
  cm_rf <- confusionMatrix(predicted_rf, test_rf$default)  
  
  cm_11[i] <- cm_rf[1,1]
  cm_12[i] <- cm_rf[1,2]
  cm_21[i] <- cm_rf[2,1]
  cm_22[i] <- cm_rf[2,2]
  
}

confmat_rf <- cbind(cm_11, cm_12, cm_21, cm_22)
confmat_rf <- data.frame(confmat_rf)
confmat_rf$accuracy <- (cm_11+cm_22)/(cm_11+cm_12+cm_21+cm_22)
confmat_rf
round(colMeans(confmat_rf),2)

hist(confmat_rf$accuracy, main="Accuracy rate of Random Forest", xlab="Accuracy")

caret::confusionMatrix(as.factor(predicted_rf), as.factor(test_rf$default))

plot(density(confmat_rf$accuracy))

#rm(list = ls())
