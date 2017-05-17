library(h2o)
library(caret)
library(cvAUC)
library(h2oEnsemble)
library(xgboost)

library(h2oEnsemble)  # This will load the `h2o` R package as well
h2o.init(nthreads = 7)  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll()

set.seed(1111)

setwd("C:/Users/MustafaErgin/Desktop/Digit Recognizer")

fileLocation <- setwd("../Digit Recognizer")
baseData <- as.data.frame(read.csv(file=paste0(fileLocation,"/train.csv"),header=TRUE))
submissionTestData <- as.data.frame(read.csv(file=paste0(fileLocation,"/test.csv"),header=TRUE))
baseData<-baseData[,colSums(baseData[])>0 ]

trainIndex <- createDataPartition(baseData$label, p = .98, 
                                   list = FALSE, 
                                  times = 1)
dataTrain <- baseData[ trainIndex,]
dataTest  <- baseData[-trainIndex,]


#Normalize the data
# preProcValues <-preProcess(dataTrain[,-1],method=c("center","scale"))
# dataTrain <- predict(preProcValues,dataTrain)
# dataTest <- predict(preProcValues,dataTest)
# submissionTestData <- predict(preProcValues,submissionTestData)



#Convert label as factor
dataTrain$label <- as.factor(dataTrain$label)
dataTest$label <- as.factor(dataTest$label)

#Convert to H2o format
train_h2o = as.h2o(dataTrain)
test_h2o = as.h2o(dataTest)
h2o_submission_test = as.h2o(submissionTestData)

#Deep learning train
h2o_model_deeplearning <- h2o.deeplearning(x=2:ncol(train_h2o) , y = 1,train_h2o,activation = "Tanh", input_dropout_ratio = 0.1
                                           ,balance_classes = TRUE, hidden = c(1000,1000,1000,1000)
                                           ,nesterov_accelerated_gradient = T, epochs = 1000,seed=1111)

#Random Forest train
h2o_model_rand<-h2o.randomForest(x=2:ncol(train_h2o) , y = 1,train_h2o,mtries=2,ntrees=1000,stopping_metric="misclassification",seed=1111)

# Naives Bayesian train
h2o_model_naiveBayes<-h2o.naiveBayes(x=2:ncol(train_h2o) , y = 1,train_h2o,compute_metrics=TRUE,seed=1111)

# k-means train
h2o_model_kmeans <- h2o.kmeans(x=2:ncol(train_h2o),train_h2o,max_iterations = 10000,k=10, standardize = TRUE,seed=1111)


# Gradient Boosting algorithm train
h2o_model_gbm<-h2o.gbm(x=2:ncol(train_h2o) , y = 1,train_h2o,ntrees=1500,seed=1111)

#save the model results
h2o.saveModel(h2o_model_deeplearning,file = "h2o_model_deeplearning.rda")
h2o.saveModel(h2o_model_gbm,file = "h2o_model_gbm.rda")
h2o.saveModel(h2o_model_rand,file = "h2o_model_rand.rda")
h2o.saveModel(h2o_model_naiveBayes,file = "h2o_model_naiveBayes.rda")
h2o.saveModel(h2o_model_kmeans,file = "h2o_model_kmeans.rda")

# Run confusion matrix for each
h2o.confusionMatrix(h2o_model_deeplearning)
h2o.confusionMatrix(h2o_model_rand)
h2o.confusionMatrix(h2o_model_gbm)
h2o.confusionMatrix(h2o_model_naiveBayes)
h2o.confusionMatrix(h2o_model_kmeans)

#Combine them and check the result on validation set
h2o_y_test_deep<- h2o.predict(h2o_model_deeplearning, test_h2o)
h2o_y_test_rand <- h2o.predict(h2o_model_rand, test_h2o)
h2o_y_test_gbm <- h2o.predict(h2o_model_gbm, test_h2o)
h2o_y_test_naive <- h2o.predict(h2o_model_naiveBayes, test_h2o)
h2o_y_test_kmeans <- h2o.predict(h2o_model_kmeans, test_h2o)

#convert them to normal R data frames
df_y_test_gbm = as.data.frame(h2o_y_test_gbm)
df_y_test_rand = as.data.frame(h2o_y_test_rand)
df_y_test_naive = as.data.frame(h2o_y_test_naive)
df_y_test_deep = as.data.frame(h2o_y_test_deep)
df_y_test_kmeans = as.data.frame(h2o_y_test_kmeans)

#Combine the models and retrain
h2o_predDF <- data.frame(df_y_test_gbm$predict,df_y_test_rand$predict,df_y_test_naive$predict,df_y_test_deep$predict,df_y_test_kmeans$predict,label=dataTest$label)
h2o_predDF = as.h2o(h2o_predDF)#convert back to H2o to use the h2o model

#Test validation data set accuracy
h2o_final_model <- h2o.gbm(x=1:(ncol(h2o_predDF)-1) , y = 6, h2o_predDF,ntrees=20000)

# Run confusion matrix for ensemble model
h2o.confusionMatrix(h2o_final_model)

#Save ensemble model
h2o.saveModel(h2o_final_model,file = "h2o_final_model.rda")

#Run the ensemble model on the submission data
submissionResults <- h2o.predict(h2o_final_model, h2o_predDF)

#Save the submission results
submission <- as.data.frame(submissionResults)
submission <- data.frame(ImageId = seq(1,length(submission$predict)), Label = submission$predict)
write.csv(submission, file = "submission.csv", row.names=F)

h2o.shutdown()
