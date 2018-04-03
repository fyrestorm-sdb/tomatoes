library(ggplot2)
library(caTools)
library(rpart)
library(rpart.plot)
library(data.table)
library(mlr)
library(stringr)
library(dplyr)
library(pROC)
library(caret)
library(xgboost)
library(precrec)
setwd("T:/deeplearning/tomato/vgg")
vg=read.csv('vg.csv')
vu=read.csv('vu.csv')
y=read.csv('y.csv')
rocc=roc( y$label,vu$pred)  #utiliser vu unripe pour estimer
plot(rocc,  print.thres = seq(0,1,by=0.1))
auc(rocc)
#Area under the curve: 0.963
threshold=coords(rocc, "best", ret = "threshold")
#[1] 0.9030647
plot(rocc,  print.thres=threshold)
th=vu$pred
vuth=vu
vuth$pred[vu$pred>threshold]=1
vuth$pred[vu$pred<threshold]=0
conf=table(cbind(vuth$pred,y))
confusionMatrix(conf)
# Confusion Matrix and Statistics

         # label
# vuth$pred  0  1
        # 0 16  1
        # 1  2 17
                                          
               # Accuracy : 0.9167          
                 # 95% CI : (0.7753, 0.9825)
    # No Information Rate : 0.5             
    # P-Value [Acc > NIR] : 1.136e-07       
                                          
                  # Kappa : 0.8333          
 # Mcnemar's Test P-Value : 1               
                                          
            # Sensitivity : 0.8889          
            # Specificity : 0.9444          
         # Pos Pred Value : 0.9412          
         # Neg Pred Value : 0.8947          
             # Prevalence : 0.5000          
         # Detection Rate : 0.4444          
   # Detection Prevalence : 0.4722          
      # Balanced Accuracy : 0.9167          
                                          
       # 'Positive' Class : 0               
recall(conf)
#[1] 0.8888889
precision(conf)
#[1] 0.9411765
F_meas(conf)
#[1] 0.9142857

sc <- evalmod(scores = vu$pred, labels = y$label)
autoplot(sc, "PRC")