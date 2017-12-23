---
title: "Logistic Regression Models"
author: "Rush"
output:
  html_document:
    keep_md: TRUE
editor_options: 
  chunk_output_type: console
---



In this file, we try multiple logistic regression models and compare them based on the values of AIC, classification accuracy, $F_1$ scores, etc.

## Data Preprocessing

Loading required libraries

```r
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyr)
library(reshape2)
library(car)
library(bestglm)
library(caret)
library(glmnet)
library(pROC)
library(SDMTools)
```

Reading the data

```r
train <- read.csv("clean_train.csv", stringsAsFactors = FALSE)
test <- read.csv("clean_test.csv", stringsAsFactors = FALSE)
```


Converting both the date variabels to their corresponding integer values, which specifies the number of days since the date 1970-01-01. This is done to make the date continuous variable. Some other variables are converted to factors.


```r
train <- train %>%
    mutate(large_avg = factor(large_avg), pur3yr = factor(pur3yr)) %>%
    mutate(lpurseason = factor(lpurseason), slscmp = factor(slscmp)) %>%
    mutate(id = as.character(id))

test <- test %>%
    mutate(large_avg = factor(large_avg), pur3yr = factor(pur3yr)) %>%
    mutate(lpurseason = factor(lpurseason), slscmp = factor(slscmp)) %>%
    mutate(id = as.character(id))
```

Removing last purchase year from the data because we have already accounted for the last purchase date in the `datelp6` variable. Now we create a test and a training data set for the logistic regression modeling process.


```r
train_logistic <- select(train, -train, -targdol, -lpuryear, -lpurseason_year)
test_logistic <- select(test, -train, -targdol, -lpuryear, -lpurseason_year)

train_linear <- train %>%
    filter(responded==1) %>%
    select(-datead6, -datelp6) %>%
    select(-train, -lpuryear, -responded, -lpurseason_year, -recency_bin) 

test_linear <- test %>%
    select(-datead6, -datelp6) %>%
    select(-train, -lpuryear, -responded, -lpurseason_year, -recency_bin)
```



## Exploratory data analysis

### Scatter plots

Plotting the response variable with the preditor variables to see if we can find any pattern


```r
ggplot(data = train, aes(x = datead6, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = datelp6, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = slstyr, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = slslyr, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = sls2ago, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = sls3ago, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = log(slshist+1), y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = log(ordtyr+1), y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = ordlyr, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = ord2ago, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = ord3ago, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = ordhist, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = lpurseason, y = targdol)) +
    geom_point()

ggplot(data = train, aes(x = ordtyr, y = targdol)) +
    geom_point()
```

### Histograms

Checking univariate distributions of the variables.

```r
qplot(x = targdol, data = train_linear, geom = "histogram")
qplot(x = datead6, data = train_linear, geom = "histogram")
qplot(x = datelp6, data = train_linear, geom = "histogram")
qplot(x = slstyr-slslyr + min(slstyr-slslyr), data = train_linear, geom = "histogram")
qplot(x = slslyr-sls2ago, data = train_linear, geom = "histogram")
qplot(x = sls2ago-sls3ago, data = train_linear, geom = "histogram")
qplot(x = sls3ago-sls4bfr, data = train_linear, geom = "histogram")
qplot(x = slshist, data = train_linear, geom = "histogram")
qplot(x = ordtyr, data = train_linear, geom = "histogram")
qplot(x = ordlyr, data = train_linear, geom = "histogram")
qplot(x = ord2ago, data = train_linear, geom = "histogram")
qplot(x = ord3ago, data = train_linear, geom = "histogram")
qplot(x = ordhist, data = train_linear, geom = "histogram")

qplot(x = falord, data = train_linear, geom = "histogram")
qplot(x = sprord, data = train_linear, geom = "histogram")
qplot(x = recency_bin, data = train_linear, geom = "histogram")
qplot(x = recency, data = train_linear, geom = "histogram")
qplot(x = lifetime, data = train_linear, geom = "histogram")
qplot(x = active, data = train_linear, geom = "histogram")
qplot(x = avg_amount, data = train_linear, geom = "histogram")
qplot(x = ord4bfr, data = train_linear, geom = "histogram")
qplot(x = sls4bfr, data = train_linear, geom = "histogram")
```

We observe that many of these distributions are skewed. Therefore, we take the log transforamtion to see if these distributions become more symmetric.

### Histograms of log transformed predictors

Checking distributions after log transformation. Adding a small displacement to avoid taking log of zeros.

A small constant to add before taking log

```r
K = 0.0001
```



```r
qplot(x = log(targdol + K), data = train, geom = "histogram")
qplot(x = log(datead6 + K), data = train, geom = "histogram")
qplot(x = log(datelp6 + K), data = train, geom = "histogram")
qplot(x = log(slstyr + K), data = train, geom = "histogram")
qplot(x = log(slslyr + K), data = train, geom = "histogram")
qplot(x = log(sls2ago + K), data = train, geom = "histogram")
qplot(x = log(sls3ago + K), data = train, geom = "histogram")
qplot(x = log(slshist + K), data = train, geom = "histogram")
qplot(x = log(ordtyr + K), data = train, geom = "histogram")
qplot(x = log(ordlyr + K), data = train, geom = "histogram")
qplot(x = log(ord2ago + K), data = train, geom = "histogram")
qplot(x = log(ord3ago + K), data = train, geom = "histogram")
qplot(x = log(ordhist + K), data = train, geom = "histogram")
qplot(x = log(falord + K), data = train, geom = "histogram")
qplot(x = log(sprord + K), data = train, geom = "histogram")
qplot(x = log(recency_bin + K), data = train, geom = "histogram")
# qplot(x = log(recency + K), data = train, geom = "histogram")
# qplot(x = log(lifetime + K), data = train, geom = "histogram")
# qplot(x = log(active + K), data = train, geom = "histogram")
qplot(x = log(avg_amount + K), data = train, geom = "histogram")
# qplot(x = log(large_avg + K), data = train, geom = "histogram")
qplot(x = log(ord4bfr + K), data = train, geom = "histogram")
qplot(x = log(sls4bfr + K), data = train, geom = "histogram")
```

## Logistic regression models

### Fitting a basic logistic regression model

A logistic regression model with all the variabels to set a base line. Some variabels are removed from the model becuase they were introducing multicollinearity and hence resulting in NA values while fitting the model.


```r
logistic1 <- glm(responded ~ . - ordhist - slshist - recency_bin - ord4bfr, data = select(train_logistic, -id, -datead6, -datelp6), 
              family = binomial)
summary(logistic1)
```

```

Call:
glm(formula = responded ~ . - ordhist - slshist - recency_bin - 
    ord4bfr, family = binomial, data = select(train_logistic, 
    -id, -datead6, -datelp6))

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-7.4854  -0.4316  -0.3220  -0.1574   4.1606  

Coefficients:
                   Estimate Std. Error z value Pr(>|z|)    
(Intercept)       1.869e+00  2.375e-01   7.869 3.58e-15 ***
slstyr            2.550e-03  5.534e-04   4.607 4.08e-06 ***
slslyr            1.262e-03  5.586e-04   2.259 0.023868 *  
sls2ago           1.088e-03  6.210e-04   1.752 0.079700 .  
sls3ago           1.312e-03  4.318e-04   3.038 0.002381 ** 
ordtyr           -1.280e-01  4.648e-02  -2.754 0.005890 ** 
ordlyr           -1.453e-01  3.986e-02  -3.646 0.000267 ***
ord2ago          -1.999e-01  3.914e-02  -5.107 3.28e-07 ***
ord3ago          -2.365e-01  3.782e-02  -6.252 4.04e-10 ***
falord            3.294e-01  1.736e-02  18.970  < 2e-16 ***
sprord            1.150e-01  1.966e-02   5.851 4.90e-09 ***
lpurseasonspring  6.482e-01  4.194e-02  15.454  < 2e-16 ***
recency          -1.361e-03  7.582e-05 -17.955  < 2e-16 ***
lifetime         -2.682e-04  2.583e-05 -10.382  < 2e-16 ***
active            2.810e+00  2.029e-01  13.852  < 2e-16 ***
avg_amount       -3.873e-03  8.772e-04  -4.415 1.01e-05 ***
large_avg1        6.527e-03  4.880e-02   0.134 0.893607    
pur3yr1          -4.960e-01  7.618e-02  -6.511 7.45e-11 ***
slscmp1          -2.469e-01  6.661e-02  -3.707 0.000209 ***
sls4bfr          -1.832e-03  2.742e-04  -6.681 2.38e-11 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 31907  on 50417  degrees of freedom
Residual deviance: 25895  on 50398  degrees of freedom
AIC: 25935

Number of Fisher Scoring iterations: 7
```

Checking the Correct Classification Rate (CCR) or the accuracy on the test data.


```
0 0.907540008608209   
0.29 0.908518214187894   
0.3 0.909789881441484   
0.31 0.911081112806667   
0.32 0.912118010721133   
0.33 0.912567985287788   
0.34 0.913037523966037   
0.35 0.913370113863129   
0.36 0.914093985992096   
0.37 0.914426575889189   
0.38 0.914993935125406   
0.39 0.915111319794968   
0.4 0.915209140352937   
```

```
[1] 0.4
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 44931   642
         1  3825  1020
                                          
               Accuracy : 0.9114          
                 95% CI : (0.9089, 0.9139)
    No Information Rate : 0.967           
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.2781          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9215          
            Specificity : 0.6137          
         Pos Pred Value : 0.9859          
         Neg Pred Value : 0.2105          
             Prevalence : 0.9670          
         Detection Rate : 0.8912          
   Detection Prevalence : 0.9039          
      Balanced Accuracy : 0.7676          
                                          
       'Positive' Class : 0               
                                          
```

Checking the CCR on the test data

```r
caret::confusionMatrix(test_logistic$responded, as.integer(pred_test>thresh))
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45737   651
         1  3683  1043
                                          
               Accuracy : 0.9152          
                 95% CI : (0.9128, 0.9176)
    No Information Rate : 0.9669          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.2903          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9255          
            Specificity : 0.6157          
         Pos Pred Value : 0.9860          
         Neg Pred Value : 0.2207          
             Prevalence : 0.9669          
         Detection Rate : 0.8948          
   Detection Prevalence : 0.9075          
      Balanced Accuracy : 0.7706          
                                          
       'Positive' Class : 0               
                                          
```

**CCR in the test data = 0.9152**

Checking the F1 score of this logistic model

```r
F_meas(factor(as.integer(pred_test>thresh)), 
       reference = as.factor(test_logistic$responded), relevant = "1")
```

```
[1] 0.3249221
```

**F1 score = 0.325**

### Fitting a logistic model with log transformed predictors


```r
logistic2 <- glm(responded ~ log(slstyr + K) + log(slslyr + K) +
                log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) +
                log(ordtyr + K) + log(ordlyr + K) +
                log(ord2ago + K) + log(ord3ago + K) + log(ord4bfr + K) +
                log(sprord+K) + log(falord+K) +
                log(avg_amount + K) + large_avg + active + lifetime + 
                recency + lpurseason, data = select(train_logistic, -id), 
        family = binomial)
summary(logistic2)
```

```

Call:
glm(formula = responded ~ log(slstyr + K) + log(slslyr + K) + 
    log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) + 
    log(ordtyr + K) + log(ordlyr + K) + log(ord2ago + K) + log(ord3ago + 
    K) + log(ord4bfr + K) + log(sprord + K) + log(falord + K) + 
    log(avg_amount + K) + large_avg + active + lifetime + recency + 
    lpurseason, family = binomial, data = select(train_logistic, 
    -id))

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.7423  -0.4138  -0.2833  -0.1890   3.3991  

Coefficients:
                      Estimate Std. Error z value Pr(>|z|)    
(Intercept)          1.778e+00  2.356e-01   7.546 4.48e-14 ***
log(slstyr + K)      1.184e-02  1.961e-02   0.604  0.54593    
log(slslyr + K)     -1.089e-02  2.500e-02  -0.436  0.66299    
log(sls2ago + K)    -5.755e-02  2.549e-02  -2.258  0.02394 *  
log(sls3ago + K)     3.890e-03  2.437e-02   0.160  0.87320    
log(sls4bfr + K)    -2.990e-01  1.125e-02 -26.576  < 2e-16 ***
log(ordtyr + K)      1.721e-02  2.716e-02   0.634  0.52636    
log(ordlyr + K)      5.317e-02  3.431e-02   1.550  0.12120    
log(ord2ago + K)     1.050e-01  3.497e-02   3.004  0.00267 ** 
log(ord3ago + K)     2.360e-02  3.339e-02   0.707  0.47961    
log(ord4bfr + K)     4.637e-01  1.466e-02  31.636  < 2e-16 ***
log(sprord + K)     -3.531e-02  5.825e-03  -6.061 1.35e-09 ***
log(falord + K)      5.801e-02  6.758e-03   8.583  < 2e-16 ***
log(avg_amount + K)  1.411e-01  2.258e-02   6.250 4.12e-10 ***
large_avg1           4.890e-03  4.547e-02   0.108  0.91437    
active               1.518e+00  2.693e-01   5.636 1.74e-08 ***
lifetime            -7.638e-05  2.671e-05  -2.859  0.00425 ** 
recency             -9.786e-04  6.758e-05 -14.481  < 2e-16 ***
lpurseasonspring     5.748e-01  5.168e-02  11.122  < 2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 31907  on 50417  degrees of freedom
Residual deviance: 24294  on 50399  degrees of freedom
AIC: 24332

Number of Fisher Scoring iterations: 6
```

Checking the CCR on the test data

```
0 0.907540008608209   
0.26 0.909965958445827   
0.27 0.912294087725476   
0.28 0.914289627108033   
0.29 0.916089525374653   
0.3 0.91740032085143   
0.31 0.918476346989083   
0.32 0.919552373126736   
0.33 0.920354501702078   
0.34 0.921078373831044   
0.35 0.922037015299135   
0.36 0.922663066870133   
0.37 0.923230426106351   
0.38 0.923699964784599   
0.39 0.923797785342568   
0.4 0.924052118793286   
0.42 0.924208631686035   
```

```
[1] 0.42
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45148   425
         1  3584  1261
                                          
               Accuracy : 0.9205          
                 95% CI : (0.9181, 0.9228)
    No Information Rate : 0.9666          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.3541          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9265          
            Specificity : 0.7479          
         Pos Pred Value : 0.9907          
         Neg Pred Value : 0.2603          
             Prevalence : 0.9666          
         Detection Rate : 0.8955          
   Detection Prevalence : 0.9039          
      Balanced Accuracy : 0.8372          
                                          
       'Positive' Class : 0               
                                          
```

Checking the CCR on the test data

```r
caret::confusionMatrix(test_logistic$responded, as.integer(pred_test>thresh))
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45999   389
         1  3485  1241
                                          
               Accuracy : 0.9242          
                 95% CI : (0.9219, 0.9265)
    No Information Rate : 0.9681          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.3602          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9296          
            Specificity : 0.7613          
         Pos Pred Value : 0.9916          
         Neg Pred Value : 0.2626          
             Prevalence : 0.9681          
         Detection Rate : 0.8999          
   Detection Prevalence : 0.9075          
      Balanced Accuracy : 0.8455          
                                          
       'Positive' Class : 0               
                                          
```

**CCR on the test data = 0.9242**

Checking the F1 score of this logistic model

```r
F_meas(factor(as.integer(pred_test>thresh)), 
       reference = as.factor(test_logistic$responded), relevant = "1")
```

```
[1] 0.3904972
```

**F1 score on the test data = 0.39**

Significant improvement from the previous model. We definitely should regress on the log of predictors.

### Adding interactions between sales

We add some interactions between sales of last 1, 2 and 3 years to capture consistency. We added these interactions one by one to see if they increase the CCR and the F1 Score on the test set. We kept all the interactions that resulted in an increase in the CCR and F1 Score. Adding similar kinds of interactions between the orders of last 1,2 or 3 years did not increase the F1 score or the CCR. Therefore, we did not add them.


```r
logistic3 <- glm(responded ~ log(slstyr + K) + log(slslyr + K) +
                log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) +
                log(ordtyr + K) + log(ordlyr + K) +
                log(ord2ago + K) + log(ord3ago + K) + log(ord4bfr + K) +
                log(sprord+K) + log(falord+K) +
                log(avg_amount + K) + large_avg + active + lifetime + 
                recency + lpurseason + 
                log(slstyr + K):log(slslyr + K) +
                log(slslyr + K):log(sls2ago + K) + 
                log(sls2ago + K):log(sls3ago + K) +
                log(sls3ago + K):log(sls4bfr + K), 
                data = select(train_logistic, -id), 
        family = binomial)
summary(logistic3)
```

```

Call:
glm(formula = responded ~ log(slstyr + K) + log(slslyr + K) + 
    log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) + 
    log(ordtyr + K) + log(ordlyr + K) + log(ord2ago + K) + log(ord3ago + 
    K) + log(ord4bfr + K) + log(sprord + K) + log(falord + K) + 
    log(avg_amount + K) + large_avg + active + lifetime + recency + 
    lpurseason + log(slstyr + K):log(slslyr + K) + log(slslyr + 
    K):log(sls2ago + K) + log(sls2ago + K):log(sls3ago + K) + 
    log(sls3ago + K):log(sls4bfr + K), family = binomial, data = select(train_logistic, 
    -id))

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.7984  -0.4161  -0.2878  -0.1899   3.9994  

Coefficients:
                                    Estimate Std. Error z value Pr(>|z|)
(Intercept)                        2.364e+00  2.625e-01   9.007  < 2e-16
log(slstyr + K)                    1.875e-02  1.975e-02   0.949  0.34254
log(slslyr + K)                    7.780e-03  2.478e-02   0.314  0.75359
log(sls2ago + K)                  -3.938e-02  2.502e-02  -1.574  0.11543
log(sls3ago + K)                   6.461e-03  2.373e-02   0.272  0.78539
log(sls4bfr + K)                  -3.056e-01  1.148e-02 -26.618  < 2e-16
log(ordtyr + K)                   -1.498e-02  2.731e-02  -0.548  0.58342
log(ordlyr + K)                    6.208e-02  3.388e-02   1.832  0.06692
log(ord2ago + K)                   1.101e-01  3.422e-02   3.219  0.00129
log(ord3ago + K)                   2.820e-02  3.244e-02   0.869  0.38467
log(ord4bfr + K)                   4.556e-01  1.480e-02  30.779  < 2e-16
log(sprord + K)                   -3.197e-02  5.891e-03  -5.426 5.75e-08
log(falord + K)                    5.947e-02  6.798e-03   8.748  < 2e-16
log(avg_amount + K)                1.828e-01  2.377e-02   7.694 1.43e-14
large_avg1                         1.622e-02  4.597e-02   0.353  0.72419
active                             2.428e+00  3.115e-01   7.793 6.55e-15
lifetime                          -1.452e-04  2.998e-05  -4.843 1.28e-06
recency                           -1.319e-03  8.302e-05 -15.888  < 2e-16
lpurseasonspring                   4.614e-01  5.299e-02   8.708  < 2e-16
log(slstyr + K):log(slslyr + K)    6.926e-03  5.546e-04  12.488  < 2e-16
log(slslyr + K):log(sls2ago + K)   4.458e-03  4.954e-04   8.998  < 2e-16
log(sls2ago + K):log(sls3ago + K)  4.648e-03  5.012e-04   9.274  < 2e-16
log(sls3ago + K):log(sls4bfr + K)  1.228e-03  5.235e-04   2.345  0.01901
                                     
(Intercept)                       ***
log(slstyr + K)                      
log(slslyr + K)                      
log(sls2ago + K)                     
log(sls3ago + K)                     
log(sls4bfr + K)                  ***
log(ordtyr + K)                      
log(ordlyr + K)                   .  
log(ord2ago + K)                  ** 
log(ord3ago + K)                     
log(ord4bfr + K)                  ***
log(sprord + K)                   ***
log(falord + K)                   ***
log(avg_amount + K)               ***
large_avg1                           
active                            ***
lifetime                          ***
recency                           ***
lpurseasonspring                  ***
log(slstyr + K):log(slslyr + K)   ***
log(slslyr + K):log(sls2ago + K)  ***
log(sls2ago + K):log(sls3ago + K) ***
log(sls3ago + K):log(sls4bfr + K) *  
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 31907  on 50417  degrees of freedom
Residual deviance: 24034  on 50395  degrees of freedom
AIC: 24080

Number of Fisher Scoring iterations: 6
```

Checking the CCR on the test data

```
0 0.907540008608209   
0.24 0.908361701295144   
0.25 0.911472395038541   
0.26 0.913839652541378   
0.27 0.915972140705091   
0.28 0.918006808310835   
0.29 0.919708886019486   
0.3 0.921391399616543   
0.31 0.922897836209258   
0.32 0.924208631686035   
0.33 0.925225965488907   
0.34 0.926047658175842   
0.35 0.926478068630903   
0.36 0.926791094416403   
0.38 0.927045427867121   
0.39 0.927456274210588   
0.4 0.927554094768557   
```

```
[1] 0.4
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45045   528
         1  3329  1516
                                          
               Accuracy : 0.9235          
                 95% CI : (0.9211, 0.9258)
    No Information Rate : 0.9595          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.4063          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9312          
            Specificity : 0.7417          
         Pos Pred Value : 0.9884          
         Neg Pred Value : 0.3129          
             Prevalence : 0.9595          
         Detection Rate : 0.8934          
   Detection Prevalence : 0.9039          
      Balanced Accuracy : 0.8364          
                                          
       'Positive' Class : 0               
                                          
```

Checking the CCR on the test data

```r
caret::confusionMatrix(test_logistic$responded, as.integer(pred_test>thresh))
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45909   479
         1  3224  1502
                                          
               Accuracy : 0.9276          
                 95% CI : (0.9253, 0.9298)
    No Information Rate : 0.9612          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.416           
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9344          
            Specificity : 0.7582          
         Pos Pred Value : 0.9897          
         Neg Pred Value : 0.3178          
             Prevalence : 0.9612          
         Detection Rate : 0.8982          
   Detection Prevalence : 0.9075          
      Balanced Accuracy : 0.8463          
                                          
       'Positive' Class : 0               
                                          
```

**CCR on the test data = 0.9276**

Checking the F1 score of this logistic model

```r
F_meas(factor(as.integer(pred_test>thresh)), 
       reference = as.factor(test_logistic$responded), relevant = "1")
```

```
[1] 0.4478903
```

**F1 score on the test data = 0.4479**

Significant improvement from the previous model. Therefore we should keep these interactions in the losgistic model.

### Adding some more interactions


```r
logistic4 <- glm(responded ~ log(slstyr + K) + log(slslyr + K) +
                log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) +
                log(ordtyr + K) + log(ordlyr + K) +
                log(ord2ago + K) + log(ord3ago + K) + log(ord4bfr + K) +
                log(sprord+K) + log(falord+K) +
                log(avg_amount + K) + large_avg + active + lifetime + 
                recency + lpurseason + 
                log(slstyr + K):log(slslyr + K) +
                log(slslyr + K):log(sls2ago + K) + 
                log(sls2ago + K):log(sls3ago + K) +
                log(sls3ago + K):log(sls4bfr + K) +
            
                recency:active + lifetime:slscmp + 
                log(slstyr + K):log(ordtyr+ K) + 
                log(slstyr + K):recency, 
                data = select(train_logistic, -id), 
        family = binomial)
summary(logistic4)
```

```

Call:
glm(formula = responded ~ log(slstyr + K) + log(slslyr + K) + 
    log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) + 
    log(ordtyr + K) + log(ordlyr + K) + log(ord2ago + K) + log(ord3ago + 
    K) + log(ord4bfr + K) + log(sprord + K) + log(falord + K) + 
    log(avg_amount + K) + large_avg + active + lifetime + recency + 
    lpurseason + log(slstyr + K):log(slslyr + K) + log(slslyr + 
    K):log(sls2ago + K) + log(sls2ago + K):log(sls3ago + K) + 
    log(sls3ago + K):log(sls4bfr + K) + recency:active + lifetime:slscmp + 
    log(slstyr + K):log(ordtyr + K) + log(slstyr + K):recency, 
    family = binomial, data = select(train_logistic, -id))

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.7526  -0.3974  -0.2933  -0.2031   3.5328  

Coefficients:
                                    Estimate Std. Error z value Pr(>|z|)
(Intercept)                       -5.261e-01  5.311e-01  -0.991 0.321861
log(slstyr + K)                   -1.744e-01  5.205e-02  -3.351 0.000805
log(slslyr + K)                   -9.677e-03  2.502e-02  -0.387 0.698859
log(sls2ago + K)                  -4.554e-02  2.503e-02  -1.820 0.068797
log(sls3ago + K)                   2.097e-03  2.390e-02   0.088 0.930088
log(sls4bfr + K)                  -3.122e-01  1.146e-02 -27.236  < 2e-16
log(ordtyr + K)                    1.402e-01  7.055e-02   1.988 0.046866
log(ordlyr + K)                    6.935e-02  3.383e-02   2.050 0.040390
log(ord2ago + K)                   1.158e-01  3.422e-02   3.382 0.000719
log(ord3ago + K)                   3.420e-02  3.268e-02   1.047 0.295291
log(ord4bfr + K)                   4.546e-01  1.462e-02  31.094  < 2e-16
log(sprord + K)                   -2.928e-02  6.003e-03  -4.878 1.07e-06
log(falord + K)                    5.491e-02  6.907e-03   7.951 1.85e-15
log(avg_amount + K)                2.090e-01  2.388e-02   8.750  < 2e-16
large_avg1                         1.142e-03  4.621e-02   0.025 0.980281
active                             6.755e+00  5.946e-01  11.361  < 2e-16
lifetime                          -1.750e-04  3.083e-05  -5.676 1.38e-08
recency                           -1.848e-04  2.152e-04  -0.859 0.390378
lpurseasonspring                   4.283e-01  5.495e-02   7.794 6.50e-15
log(slstyr + K):log(slslyr + K)    6.250e-03  5.913e-04  10.570  < 2e-16
log(slslyr + K):log(sls2ago + K)   4.577e-03  5.017e-04   9.122  < 2e-16
log(sls2ago + K):log(sls3ago + K)  5.005e-03  5.079e-04   9.855  < 2e-16
log(sls3ago + K):log(sls4bfr + K)  2.536e-03  5.504e-04   4.609 4.05e-06
active:recency                    -1.414e-03  1.733e-04  -8.159 3.39e-16
lifetime:slscmp1                  -4.379e-05  1.547e-05  -2.831 0.004642
log(slstyr + K):log(ordtyr + K)    1.911e-02  7.293e-03   2.621 0.008775
log(slstyr + K):recency            9.120e-05  2.163e-05   4.216 2.49e-05
                                     
(Intercept)                          
log(slstyr + K)                   ***
log(slslyr + K)                      
log(sls2ago + K)                  .  
log(sls3ago + K)                     
log(sls4bfr + K)                  ***
log(ordtyr + K)                   *  
log(ordlyr + K)                   *  
log(ord2ago + K)                  ***
log(ord3ago + K)                     
log(ord4bfr + K)                  ***
log(sprord + K)                   ***
log(falord + K)                   ***
log(avg_amount + K)               ***
large_avg1                           
active                            ***
lifetime                          ***
recency                              
lpurseasonspring                  ***
log(slstyr + K):log(slslyr + K)   ***
log(slslyr + K):log(sls2ago + K)  ***
log(sls2ago + K):log(sls3ago + K) ***
log(sls3ago + K):log(sls4bfr + K) ***
active:recency                    ***
lifetime:slscmp1                  ** 
log(slstyr + K):log(ordtyr + K)   ** 
log(slstyr + K):recency           ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 31907  on 50417  degrees of freedom
Residual deviance: 23939  on 50391  degrees of freedom
AIC: 23993

Number of Fisher Scoring iterations: 6
```

Checking the CCR on the train data

```
0 0.907540008608209   
0.25 0.910102907226983   
0.26 0.912744062292131   
0.27 0.915033063348593   
0.28 0.917165551512306   
0.29 0.919317603787612   
0.3 0.920530578706421   
0.31 0.922076143522323   
0.32 0.923171733771569   
0.33 0.924580349806315   
0.34 0.925441170716438   
0.35 0.926184606956998   
0.36 0.926771530304809   
0.37 0.927417145987401   
0.38 0.9277301717729   
0.39 0.928141018116367   
0.4 0.928238838674336   
0.41 0.928317095120711   
```

```
[1] 0.41
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45050   523
         1  3282  1563
                                          
               Accuracy : 0.9245          
                 95% CI : (0.9222, 0.9268)
    No Information Rate : 0.9586          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.4173          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9321          
            Specificity : 0.7493          
         Pos Pred Value : 0.9885          
         Neg Pred Value : 0.3226          
             Prevalence : 0.9586          
         Detection Rate : 0.8935          
   Detection Prevalence : 0.9039          
      Balanced Accuracy : 0.8407          
                                          
       'Positive' Class : 0               
                                          
```

Checking the CCR on the test data

```r
caret::confusionMatrix(test_logistic$responded, as.integer(pred_test>thresh))
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45913   475
         1  3189  1537
                                         
               Accuracy : 0.9283         
                 95% CI : (0.926, 0.9305)
    No Information Rate : 0.9606         
    P-Value [Acc > NIR] : 1              
                                         
                  Kappa : 0.4244         
 Mcnemar's Test P-Value : <2e-16         
                                         
            Sensitivity : 0.9351         
            Specificity : 0.7639         
         Pos Pred Value : 0.9898         
         Neg Pred Value : 0.3252         
             Prevalence : 0.9606         
         Detection Rate : 0.8982         
   Detection Prevalence : 0.9075         
      Balanced Accuracy : 0.8495         
                                         
       'Positive' Class : 0              
                                         
```

**CCR on the test data = 0.9283**

Checking the F1 score of this logistic model

```r
F_meas(factor(as.integer(pred_test>thresh)), 
       reference = as.factor(test_logistic$responded), relevant = "1")
```

```
[1] 0.4562185
```

**F1 score on the test data = 0.4562**

Significant improvement from the previous model. Therefore we should keep these interactions in the losgistic model.


### Adding some more interactions

Adding these interactions increased the CCR but resulted in a decrease in the F1 score.


```r
logistic5 <- glm(responded ~ log(slstyr + K) + log(slslyr + K) +
                log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) +
                log(ordtyr + K) + log(ordlyr + K) +
                log(ord2ago + K) + log(ord3ago + K) + log(ord4bfr + K) +
                log(sprord+K) + log(falord+K) +
                log(avg_amount + K) + large_avg + active + lifetime + 
                recency + lpurseason + 
                log(slstyr + K):log(slslyr + K) +
                log(slslyr + K):log(sls2ago + K) + 
                log(sls2ago + K):log(sls3ago + K) +
                log(sls3ago + K):log(sls4bfr + K) +
            
                recency:active + lifetime:slscmp + 
                log(slstyr + K):log(ordtyr+ K) + 
                log(slstyr + K):recency + 
                    
                    log(slslyr + K):recency +
                    log(sls2ago + K):log(ord2ago + K) + 
                    log(avg_amount + K):recency, 
                data = select(train_logistic, -id), 
        family = binomial)
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
summary(logistic5)
```

```

Call:
glm(formula = responded ~ log(slstyr + K) + log(slslyr + K) + 
    log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) + 
    log(ordtyr + K) + log(ordlyr + K) + log(ord2ago + K) + log(ord3ago + 
    K) + log(ord4bfr + K) + log(sprord + K) + log(falord + K) + 
    log(avg_amount + K) + large_avg + active + lifetime + recency + 
    lpurseason + log(slstyr + K):log(slslyr + K) + log(slslyr + 
    K):log(sls2ago + K) + log(sls2ago + K):log(sls3ago + K) + 
    log(sls3ago + K):log(sls4bfr + K) + recency:active + lifetime:slscmp + 
    log(slstyr + K):log(ordtyr + K) + log(slstyr + K):recency + 
    log(slslyr + K):recency + log(sls2ago + K):log(ord2ago + 
    K) + log(avg_amount + K):recency, family = binomial, data = select(train_logistic, 
    -id))

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-3.3874  -0.3979  -0.2935  -0.2095   4.8644  

Coefficients:
                                    Estimate Std. Error z value Pr(>|z|)
(Intercept)                        3.885e+00  6.238e-01   6.228 4.73e-10
log(slstyr + K)                   -2.347e-01  4.781e-02  -4.909 9.15e-07
log(slslyr + K)                    4.551e-01  5.673e-02   8.023 1.03e-15
log(sls2ago + K)                  -2.351e-02  2.632e-02  -0.893 0.371785
log(sls3ago + K)                   1.030e-02  2.476e-02   0.416 0.677524
log(sls4bfr + K)                  -3.091e-01  1.180e-02 -26.200  < 2e-16
log(ordtyr + K)                    7.204e-02  7.219e-02   0.998 0.318263
log(ordlyr + K)                    5.638e-03  3.624e-02   0.156 0.876356
log(ord2ago + K)                   3.196e-01  6.952e-02   4.597 4.29e-06
log(ord3ago + K)                   2.298e-02  3.386e-02   0.679 0.497418
log(ord4bfr + K)                   4.488e-01  1.509e-02  29.731  < 2e-16
log(sprord + K)                   -2.941e-02  6.045e-03  -4.865 1.14e-06
log(falord + K)                    5.342e-02  6.992e-03   7.641 2.16e-14
log(avg_amount + K)               -4.440e-01  1.028e-01  -4.320 1.56e-05
large_avg1                         7.832e-02  4.763e-02   1.644 0.100133
active                             6.474e+00  5.848e-01  11.069  < 2e-16
lifetime                          -1.713e-04  3.106e-05  -5.516 3.46e-08
recency                           -2.021e-03  2.522e-04  -8.014 1.11e-15
lpurseasonspring                   3.568e-01  5.575e-02   6.401 1.54e-10
log(slstyr + K):log(slslyr + K)    1.057e-03  8.193e-04   1.290 0.197050
log(slslyr + K):log(sls2ago + K)   4.182e-03  5.073e-04   8.245  < 2e-16
log(sls2ago + K):log(sls3ago + K)  4.776e-03  5.123e-04   9.324  < 2e-16
log(sls3ago + K):log(sls4bfr + K)  2.586e-03  5.534e-04   4.674 2.96e-06
active:recency                    -1.309e-03  1.664e-04  -7.868 3.61e-15
lifetime:slscmp1                  -4.208e-05  1.558e-05  -2.701 0.006916
log(slstyr + K):log(ordtyr + K)    2.236e-02  7.336e-03   3.048 0.002304
log(slstyr + K):recency            1.339e-04  1.814e-05   7.382 1.56e-13
log(slslyr + K):recency           -1.806e-04  2.109e-05  -8.564  < 2e-16
log(sls2ago + K):log(ord2ago + K)  2.598e-02  6.835e-03   3.802 0.000144
log(avg_amount + K):recency        1.960e-04  3.233e-05   6.063 1.33e-09
                                     
(Intercept)                       ***
log(slstyr + K)                   ***
log(slslyr + K)                   ***
log(sls2ago + K)                     
log(sls3ago + K)                     
log(sls4bfr + K)                  ***
log(ordtyr + K)                      
log(ordlyr + K)                      
log(ord2ago + K)                  ***
log(ord3ago + K)                     
log(ord4bfr + K)                  ***
log(sprord + K)                   ***
log(falord + K)                   ***
log(avg_amount + K)               ***
large_avg1                           
active                            ***
lifetime                          ***
recency                           ***
lpurseasonspring                  ***
log(slstyr + K):log(slslyr + K)      
log(slslyr + K):log(sls2ago + K)  ***
log(sls2ago + K):log(sls3ago + K) ***
log(sls3ago + K):log(sls4bfr + K) ***
active:recency                    ***
lifetime:slscmp1                  ** 
log(slstyr + K):log(ordtyr + K)   ** 
log(slstyr + K):recency           ***
log(slslyr + K):recency           ***
log(sls2ago + K):log(ord2ago + K) ***
log(avg_amount + K):recency       ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 31907  on 50417  degrees of freedom
Residual deviance: 23814  on 50388  degrees of freedom
AIC: 23874

Number of Fisher Scoring iterations: 7
```

Checking the CCR on the train data

```
0 0.907540008608209   
0.24 0.908185624290801   
0.25 0.911139805141449   
0.26 0.913370113863129   
0.27 0.915933012481903   
0.28 0.917967680087647   
0.29 0.920021911804985   
0.3 0.921704425402042   
0.31 0.923210861994757   
0.32 0.924110811128067   
0.33 0.92503032437297   
0.34 0.926341119849748   
0.35 0.927064991978714   
0.36 0.927612787103338   
0.37 0.927769299996087   
0.38 0.928317095120711   
0.39 0.928571428571429   
0.4 0.928590992683022   
0.41 0.928688813240991   
0.42 0.928708377352584   
0.44 0.928806197910553   
0.45 0.928825762022147   
0.46 0.928943146691709   
```

```
[1] 0.46
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 45178   395
         1  3385  1460
                                          
               Accuracy : 0.925           
                 95% CI : (0.9227, 0.9273)
    No Information Rate : 0.9632          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.4041          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9303          
            Specificity : 0.7871          
         Pos Pred Value : 0.9913          
         Neg Pred Value : 0.3013          
             Prevalence : 0.9632          
         Detection Rate : 0.8961          
   Detection Prevalence : 0.9039          
      Balanced Accuracy : 0.8587          
                                          
       'Positive' Class : 0               
                                          
```

Checking the CCR on the test data

```r
caret::confusionMatrix(test_logistic$responded, as.integer(pred_test>thresh))
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 46030   358
         1  3274  1452
                                          
               Accuracy : 0.9289          
                 95% CI : (0.9267, 0.9312)
    No Information Rate : 0.9646          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.4143          
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9336          
            Specificity : 0.8022          
         Pos Pred Value : 0.9923          
         Neg Pred Value : 0.3072          
             Prevalence : 0.9646          
         Detection Rate : 0.9005          
   Detection Prevalence : 0.9075          
      Balanced Accuracy : 0.8679          
                                          
       'Positive' Class : 0               
                                          
```

**CCR on the test data = 0.9289**

Checking the F1 score of this logistic model

```r
F_meas(factor(as.integer(pred_test>thresh)), 
       reference = as.factor(test_logistic$responded), relevant = "1")
```

```
[1] 0.4443084
```

**F1 score on the test data = 0.4443**

### Backwards stepwise on logistic5 


```
Start:  AIC=23874.12
responded ~ log(slstyr + K) + log(slslyr + K) + log(sls2ago + 
    K) + log(sls3ago + K) + log(sls4bfr + K) + log(ordtyr + K) + 
    log(ordlyr + K) + log(ord2ago + K) + log(ord3ago + K) + log(ord4bfr + 
    K) + log(sprord + K) + log(falord + K) + log(avg_amount + 
    K) + large_avg + active + lifetime + recency + lpurseason + 
    log(slstyr + K):log(slslyr + K) + log(slslyr + K):log(sls2ago + 
    K) + log(sls2ago + K):log(sls3ago + K) + log(sls3ago + K):log(sls4bfr + 
    K) + recency:active + lifetime:slscmp + log(slstyr + K):log(ordtyr + 
    K) + log(slstyr + K):recency + log(slslyr + K):recency + 
    log(sls2ago + K):log(ord2ago + K) + log(avg_amount + K):recency
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
                                    Df Deviance   AIC
- log(ordlyr + K)                    1    23814 23872
- log(ord3ago + K)                   1    23815 23873
- log(slstyr + K):log(slslyr + K)    1    23816 23874
<none>                                    23814 23874
- large_avg                          1    23817 23875
- lifetime:slscmp                    1    23821 23879
- log(slstyr + K):log(ordtyr + K)    1    23823 23881
- log(sls2ago + K):log(ord2ago + K)  1    23828 23886
- log(sls3ago + K):log(sls4bfr + K)  1    23836 23894
- log(sprord + K)                    1    23838 23896
- log(slstyr + K):recency            1    23855 23913
- lpurseason                         1    23856 23914
- log(avg_amount + K):recency        1    23857 23915
- log(falord + K)                    1    23874 23932
- active:recency                     1    23876 23934
- log(slslyr + K):log(sls2ago + K)   1    23882 23940
- log(slslyr + K):recency            1    23889 23947
- log(sls2ago + K):log(sls3ago + K)  1    23902 23960
- log(ord4bfr + K)                   1    25345 25403
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```

Step:  AIC=23872.15
responded ~ log(slstyr + K) + log(slslyr + K) + log(sls2ago + 
    K) + log(sls3ago + K) + log(sls4bfr + K) + log(ordtyr + K) + 
    log(ord2ago + K) + log(ord3ago + K) + log(ord4bfr + K) + 
    log(sprord + K) + log(falord + K) + log(avg_amount + K) + 
    large_avg + active + lifetime + recency + lpurseason + log(slstyr + 
    K):log(slslyr + K) + log(slslyr + K):log(sls2ago + K) + log(sls2ago + 
    K):log(sls3ago + K) + log(sls3ago + K):log(sls4bfr + K) + 
    active:recency + lifetime:slscmp + log(slstyr + K):log(ordtyr + 
    K) + log(slstyr + K):recency + log(slslyr + K):recency + 
    log(sls2ago + K):log(ord2ago + K) + log(avg_amount + K):recency
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
                                    Df Deviance   AIC
- log(ord3ago + K)                   1    23815 23871
- log(slstyr + K):log(slslyr + K)    1    23816 23872
<none>                                    23814 23872
- large_avg                          1    23817 23873
+ log(ordlyr + K)                    1    23814 23874
- lifetime:slscmp                    1    23821 23877
- log(slstyr + K):log(ordtyr + K)    1    23823 23879
- log(sls2ago + K):log(ord2ago + K)  1    23828 23884
- log(sls3ago + K):log(sls4bfr + K)  1    23836 23892
- log(sprord + K)                    1    23838 23894
- log(slstyr + K):recency            1    23855 23911
- lpurseason                         1    23856 23912
- log(avg_amount + K):recency        1    23860 23916
- log(falord + K)                    1    23875 23931
- active:recency                     1    23876 23932
- log(slslyr + K):log(sls2ago + K)   1    23882 23938
- log(slslyr + K):recency            1    23889 23945
- log(sls2ago + K):log(sls3ago + K)  1    23902 23958
- log(ord4bfr + K)                   1    25373 25429
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```

Step:  AIC=23870.59
responded ~ log(slstyr + K) + log(slslyr + K) + log(sls2ago + 
    K) + log(sls3ago + K) + log(sls4bfr + K) + log(ordtyr + K) + 
    log(ord2ago + K) + log(ord4bfr + K) + log(sprord + K) + log(falord + 
    K) + log(avg_amount + K) + large_avg + active + lifetime + 
    recency + lpurseason + log(slstyr + K):log(slslyr + K) + 
    log(slslyr + K):log(sls2ago + K) + log(sls2ago + K):log(sls3ago + 
    K) + log(sls3ago + K):log(sls4bfr + K) + active:recency + 
    lifetime:slscmp + log(slstyr + K):log(ordtyr + K) + log(slstyr + 
    K):recency + log(slslyr + K):recency + log(sls2ago + K):log(ord2ago + 
    K) + log(avg_amount + K):recency
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
                                    Df Deviance   AIC
- log(slstyr + K):log(slslyr + K)    1    23816 23870
<none>                                    23815 23871
- large_avg                          1    23817 23871
+ log(ord3ago + K)                   1    23814 23872
+ log(ordlyr + K)                    1    23815 23873
- lifetime:slscmp                    1    23822 23876
- log(slstyr + K):log(ordtyr + K)    1    23824 23878
- log(sls2ago + K):log(ord2ago + K)  1    23829 23883
- log(sls3ago + K):log(sls4bfr + K)  1    23837 23891
- log(sprord + K)                    1    23838 23892
- log(slstyr + K):recency            1    23856 23910
- lpurseason                         1    23856 23910
- log(avg_amount + K):recency        1    23860 23914
- log(falord + K)                    1    23875 23929
- active:recency                     1    23876 23930
- log(slslyr + K):log(sls2ago + K)   1    23883 23937
- log(slslyr + K):recency            1    23890 23944
- log(sls2ago + K):log(sls3ago + K)  1    23902 23956
- log(ord4bfr + K)                   1    25374 25428
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```

Step:  AIC=23870.21
responded ~ log(slstyr + K) + log(slslyr + K) + log(sls2ago + 
    K) + log(sls3ago + K) + log(sls4bfr + K) + log(ordtyr + K) + 
    log(ord2ago + K) + log(ord4bfr + K) + log(sprord + K) + log(falord + 
    K) + log(avg_amount + K) + large_avg + active + lifetime + 
    recency + lpurseason + log(slslyr + K):log(sls2ago + K) + 
    log(sls2ago + K):log(sls3ago + K) + log(sls3ago + K):log(sls4bfr + 
    K) + active:recency + lifetime:slscmp + log(slstyr + K):log(ordtyr + 
    K) + log(slstyr + K):recency + log(slslyr + K):recency + 
    log(sls2ago + K):log(ord2ago + K) + log(avg_amount + K):recency
```

```
Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```
                                    Df Deviance   AIC
<none>                                    23816 23870
+ log(slstyr + K):log(slslyr + K)    1    23815 23871
- large_avg                          1    23819 23871
+ log(ord3ago + K)                   1    23816 23872
+ log(ordlyr + K)                    1    23816 23872
- log(slstyr + K):log(ordtyr + K)    1    23826 23878
- lifetime:slscmp                    1    23826 23878
- log(sls2ago + K):log(ord2ago + K)  1    23830 23882
- log(sls3ago + K):log(sls4bfr + K)  1    23838 23890
- log(sprord + K)                    1    23840 23892
- lpurseason                         1    23857 23909
- log(slstyr + K):recency            1    23860 23912
- log(avg_amount + K):recency        1    23865 23917
- log(falord + K)                    1    23877 23929
- active:recency                     1    23878 23930
- log(slslyr + K):log(sls2ago + K)   1    23883 23935
- log(sls2ago + K):log(sls3ago + K)  1    23902 23954
- log(slslyr + K):recency            1    23988 24040
- log(ord4bfr + K)                   1    25377 25429
```

```

Call:
glm(formula = responded ~ log(slstyr + K) + log(slslyr + K) + 
    log(sls2ago + K) + log(sls3ago + K) + log(sls4bfr + K) + 
    log(ordtyr + K) + log(ord2ago + K) + log(ord4bfr + K) + log(sprord + 
    K) + log(falord + K) + log(avg_amount + K) + large_avg + 
    active + lifetime + recency + lpurseason + log(slslyr + K):log(sls2ago + 
    K) + log(sls2ago + K):log(sls3ago + K) + log(sls3ago + K):log(sls4bfr + 
    K) + active:recency + lifetime:slscmp + log(slstyr + K):log(ordtyr + 
    K) + log(slstyr + K):recency + log(slslyr + K):recency + 
    log(sls2ago + K):log(ord2ago + K) + log(avg_amount + K):recency, 
    family = binomial, data = select(train_logistic, -id))

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-3.4099  -0.3980  -0.2931  -0.2099   4.9671  

Coefficients:
                                    Estimate Std. Error z value Pr(>|z|)
(Intercept)                        4.128e+00  5.796e-01   7.122 1.06e-12
log(slstyr + K)                   -2.412e-01  4.716e-02  -5.114 3.16e-07
log(slslyr + K)                    5.003e-01  3.767e-02  13.281  < 2e-16
log(sls2ago + K)                  -2.269e-02  2.628e-02  -0.863 0.388070
log(sls3ago + K)                   2.719e-02  3.747e-03   7.256 4.00e-13
log(sls4bfr + K)                  -3.084e-01  1.175e-02 -26.246  < 2e-16
log(ordtyr + K)                    7.053e-02  7.186e-02   0.982 0.326290
log(ord2ago + K)                   3.191e-01  6.955e-02   4.588 4.48e-06
log(ord4bfr + K)                   4.479e-01  1.503e-02  29.804  < 2e-16
log(sprord + K)                   -2.927e-02  6.035e-03  -4.850 1.24e-06
log(falord + K)                    5.373e-02  6.980e-03   7.697 1.39e-14
log(avg_amount + K)               -4.667e-01  9.736e-02  -4.793 1.64e-06
large_avg1                         7.549e-02  4.718e-02   1.600 0.109626
active                             6.495e+00  5.842e-01  11.119  < 2e-16
lifetime                          -1.686e-04  3.096e-05  -5.445 5.18e-08
recency                           -2.148e-03  2.339e-04  -9.181  < 2e-16
lpurseasonspring                   3.544e-01  5.576e-02   6.356 2.07e-10
log(slslyr + K):log(sls2ago + K)   4.063e-03  4.987e-04   8.146 3.75e-16
log(sls2ago + K):log(sls3ago + K)  4.686e-03  5.079e-04   9.227  < 2e-16
log(sls3ago + K):log(sls4bfr + K)  2.554e-03  5.529e-04   4.620 3.84e-06
active:recency                    -1.322e-03  1.663e-04  -7.952 1.83e-15
lifetime:slscmp1                  -4.688e-05  1.507e-05  -3.110 0.001872
log(slstyr + K):log(ordtyr + K)    2.283e-02  7.314e-03   3.121 0.001802
log(slstyr + K):recency            1.372e-04  1.787e-05   7.678 1.62e-14
log(slslyr + K):recency           -1.992e-04  1.552e-05 -12.834  < 2e-16
log(sls2ago + K):log(ord2ago + K)  2.602e-02  6.835e-03   3.807 0.000141
log(avg_amount + K):recency        2.013e-04  3.123e-05   6.447 1.14e-10
                                     
(Intercept)                       ***
log(slstyr + K)                   ***
log(slslyr + K)                   ***
log(sls2ago + K)                     
log(sls3ago + K)                  ***
log(sls4bfr + K)                  ***
log(ordtyr + K)                      
log(ord2ago + K)                  ***
log(ord4bfr + K)                  ***
log(sprord + K)                   ***
log(falord + K)                   ***
log(avg_amount + K)               ***
large_avg1                           
active                            ***
lifetime                          ***
recency                           ***
lpurseasonspring                  ***
log(slslyr + K):log(sls2ago + K)  ***
log(sls2ago + K):log(sls3ago + K) ***
log(sls3ago + K):log(sls4bfr + K) ***
active:recency                    ***
lifetime:slscmp1                  ** 
log(slstyr + K):log(ordtyr + K)   ** 
log(slstyr + K):recency           ***
log(slslyr + K):recency           ***
log(sls2ago + K):log(ord2ago + K) ***
log(avg_amount + K):recency       ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 31907  on 50417  degrees of freedom
Residual deviance: 23816  on 50391  degrees of freedom
AIC: 23870

Number of Fisher Scoring iterations: 7
```

```
0 0.907540008608209   
0.24 0.907833470282114   
0.25 0.911296318034198   
0.26 0.913604883202254   
0.27 0.915815627812341   
0.28 0.917967680087647   
0.29 0.919884963023829   
0.3 0.921900066517979   
0.31 0.923054349102007   
0.32 0.923895605900536   
0.33 0.92518683726572   
0.34 0.926125914622217   
0.35 0.926888914974371   
0.36 0.92757365888015   
0.37 0.927964941112024   
0.38 0.928317095120711   
0.39 0.928551864459835   
0.4 0.928610556794616   
0.41 0.928767069687365   
0.45 0.928884454356928   
0.46 0.928943146691709   
```

```
[1] 0.46
```

```
Confusion Matrix and Statistics

          Reference
Prediction     0     1
         0 46032   356
         1  3276  1450
                                          
               Accuracy : 0.9289          
                 95% CI : (0.9267, 0.9312)
    No Information Rate : 0.9647          
    P-Value [Acc > NIR] : 1               
                                          
                  Kappa : 0.414           
 Mcnemar's Test P-Value : <2e-16          
                                          
            Sensitivity : 0.9336          
            Specificity : 0.8029          
         Pos Pred Value : 0.9923          
         Neg Pred Value : 0.3068          
             Prevalence : 0.9647          
         Detection Rate : 0.9006          
   Detection Prevalence : 0.9075          
      Balanced Accuracy : 0.8682          
                                          
       'Positive' Class : 0               
                                          
```

```
[1] 0.4439682
```


### Summary of all the logistic regression models

The CCR/accuracies and F1 scores of all the logistic regression models are given below:

- logistic1: CCR = 0.9152, F1 Score = 0.325 AIC = 25934
- logistic2: CCR = 0.9242, F1 Score = 0.39 AIC = 24332
- logistic3: CCR = 0.9276, F1 Score = 0.4479, AIC = 24080 
- logistic4: CCR = 0.9283, F1 Score = 0.4613, AIC = 23992
- logistic5: CCR = 0.9289, F1 Score = 0.4443, AIC = 23873
- (stepwise on logistic5) bwd_log: CCR = 0.929, F1 Score = 0.4439, AIC = 23870 

3 predictors removed from logistic5 after doing stepwise regression


### Writing the predicted probabilites to a file


```r
prob_test <- data.frame(id = test_logistic$id, prob = pred_test)
write.csv(prob_test, "probabilities.csv", row.names=FALSE)
```

