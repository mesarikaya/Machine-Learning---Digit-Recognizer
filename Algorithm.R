setwd("C:/Users/MustafaErgin/Desktop/Digit Recognizer")

fileLocation <- setwd("../Digit Recognizer")
baseData <- as.data.frame(read.csv(file=paste0(fileLocation,"/train.csv"),header=TRUE))

library(randomForest)
library(nnet)
library(caret)
library(caretEnsemble)
library(kknn)
nzv <- nearZeroVar(baseData)
filteredDescr <- baseData[, -nzv]
dim(filteredDescr)

set.seed(1111)
trainIndex <- createDataPartition(filteredDescr$label, p = .9, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- filteredDescr[ trainIndex,]
dataTest  <- filteredDescr[-trainIndex,]

#Convert label as factor
dataTrain$label <- as.factor(dataTrain$label)
dataTest$label <- as.factor(dataTest$label)

#Normalize the data
# preProcValues <-preProcess(dataTrain,method=c("center","scale"))
# dataTrain <- predict(preProcValues,dataTrain)
# dataTest <- predict(preProcValues,dataTest)

my.grid <- expand.grid(.decay =0.5,.size = c(100))
neuralnetGrid <- expand.grid(.layer1=100,.layer2=100,.layer3=100)
# fitControl<-trainControl(method = "repeatedcv",
#              number = 1,
#              repeats = 1,
#              search = "grid",
#              verboseIter = TRUE,
#              classProbs = TRUE,
#              summaryFunction = defaultSummary,
#              selectionFunction = "best",
#              seeds = 3456,
#              trim = FALSE,
#              allowParallel = TRUE
#              )
# 
# mtry <- sqrt(ncol(filteredDescr))
# tunegrid <- expand.grid(.mtry=mtry)
# fitControl <- trainControl( method = "repeatedcv",
#                             number = 10,
#                             ## repeated ten times
#                             repeats = 3,
#                             verboseIter = TRUE,
#                             returnData = TRUE,
#                             classProbs = TRUE,
#                             ## Evaluate performance using
#                             ## the following function
#                             summaryFunction = twoClassSummary)

fitControl <- trainControl( method="repeatedcv",
                            savePredictions =TRUE,
                            number = 10,
                            repeats=3,
                            ## repeated ten times
                            returnData = TRUE,
                            ## Evaluate performance using
                            ## the following function
                            summaryFunction = multiClassSummary,
                            seed=1111)
# my.grid <- expand.grid(.decay = 0.1,.size = c(20))

features <- colnames(dataTrain[,-which(colnames(dataTrain)=="label")])
# modelFit4 <- train(label~.,method = "mlpML",data=dataTrain,trainControl=fitControl,preProcess =c("center","scale","pca"),tuneGrid=neuralnetGrid)
# save(modelFit4,file = "modelFit4.rda")
# modelFit1 <- train(x=as.matrix(dataTrain[,features]) , y = dataTrain$label,method = "nnet", trainControl=fitControl,preProcess =c("center","scale","pca"), maxit = 200, tuneGrid = my.grid, trace = T,  MaxNWts = 50000)
# save(modelFit1,file = "modelFit1.rda")
# modelFit2 <- train(x=as.matrix(dataTrain[,features]) , y = dataTrain$label,method = "parRF",trainControl=fitControl,preProcess =c("center","scale","pca"),ntree=1500,importance=TRUE)
# save(modelFit2,file = "modelFit2.rda")
modelFit3 <- train(label~.,method = "kknn",data=dataTrain,trainControl=fitControl,preProcess =c("center","scale","pca"))
save(modelFit3,file = "modelFit3.rda")

# pred1 <- predict(modelFit1,dataTest)
# pred2 <- predict(modelFit2,dataTest)
pred3 <- predict(modelFit3,dataTest)
# pred4 <- predict(modelFit4,dataTest)
# predDF <- data.frame(pred1,pred2,pred3,pred4,label=dataTest$label)
# my.grid2 <- expand.grid(.decay =seq(from=0.1,to=1,by=0.1),.size = c(2,3,4,5,6,7,8,9,10))
# combModFit <- train(x=as.matrix(predDF[,c(1:3)]) , y = predDF$label,method = "nnet", trainControl=fitControl,maxit = 400, tuneGrid = my.grid, trace = T,  MaxNWts = 20000)
# save(combModFit,file = "CombModFit.rda")
# combPred <- predict(combModFit,predDF)

# models <- caretList(x=as.matrix(dataTrain[,features]), y = dataTrain$label, 
#                     tuneList=list(nnet=caretModelSpec(method="nnet",preProcess =c("center","scale","pca"), maxit = 100, tuneGrid = my.grid, trace = T,  MaxNWts = 10000),
#                                   rf=caretModelSpec(method="parRF",preProcess =c("center","scale","pca"),ntree=500,importance=TRUE)), 
#                     trControl = fitControl)
# 
# stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
# stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
# print(stack.glm)
# modelFit8 <- train(label~. , data=dataTrain,method = "multinom",trace=TRUE)
# modelFit6 <- train(x=as.matrix(dataTrain[,features]) , y = dataTrain$label,method = "parRF",trainControl=fitControl,tunegrid=tunegrid ,ntree=500,importance=TRUE)
